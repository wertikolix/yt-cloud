#!/usr/bin/env python3
"""
yt-cloud coder — стеганографическое хранение файлов в видео.
Кодирует произвольные файлы в цветовые блоки на видеокадрах,
устойчивые к YouTube-сжатию. Декодирует обратно без потерь.
"""

import argparse
import cv2
import hashlib
import math
import numpy as np
import os
import re
import struct
import subprocess
import sys
import tempfile
import shutil
from io import BytesIO
from tqdm import tqdm

# ─── Опционально: AES-256-GCM ───────────────────────────────────────────────
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

# ═════════════════════════════════════════════════════════════════════════════
#  ПАЛИТРА — 16 цветов, максимально разнесённых в YUV-пространстве
# ═════════════════════════════════════════════════════════════════════════════
# BGR-формат (OpenCV native).
# Выбраны так, чтобы после YUV420 chroma subsampling расстояния оставались
# максимальными. Чёрный (0,0,0) НЕ используется — это фон.

PALETTE_BGR = [
    (0,   0,   255),   # 0  — красный
    (0,   255, 0),     # 1  — зелёный
    (255, 0,   0),     # 2  — синий
    (0,   255, 255),   # 3  — жёлтый
    (255, 0,   255),   # 4  — маджента
    (255, 255, 0),     # 5  — циан
    (255, 255, 255),   # 6  — белый
    (0,   128, 255),   # 7  — оранжевый
    (128, 0,   255),   # 8  — розовый/фиолетовый
    (0,   255, 128),   # 9  — лайм
    (255, 128, 0),     # 10 — голубой (светлый)
    (128, 255, 0),     # 11 — бирюза
    (0,   0,   128),   # 12 — тёмно-красный
    (0,   128, 0),     # 13 — тёмно-зелёный
    (128, 0,   0),     # 14 — тёмно-синий
    (128, 128, 128),   # 15 — серый
]

PALETTE_NP = np.array(PALETTE_BGR, dtype=np.int32)  # для быстрого nearest-neighbor

# ═════════════════════════════════════════════════════════════════════════════
#  КОНСТАНТЫ ФОРМАТА
# ═════════════════════════════════════════════════════════════════════════════

WIDTH = 1920
HEIGHT = 1080
FPS = 6

BLOCK_W = 40         # Ширина блока (px) — крупнее = устойчивее к сжатию
BLOCK_H = 40         # Высота блока
SPACING = 4          # Промежуток между блоками (чёрный, визуальный разделитель)
MARKER_SIZE = 0      # Убраны corner-маркеры — бесполезны для декодирования

# Сетка данных
STEP_X = BLOCK_W + SPACING
STEP_Y = BLOCK_H + SPACING
GRID_COLS = WIDTH // STEP_X        # сколько блоков по X
GRID_ROWS = HEIGHT // STEP_Y       # сколько блоков по Y

# Первые 2 строки блоков — служебные (номер кадра + CRC кадра).
# Номер кадра: 4 блока = 4*4 = 16 бит = до 65535 кадров.
# CRC кадра: 4 блока = 16 бит CRC-16 данных на кадре.
HEADER_BLOCKS = 8  # 4 frame_num + 4 crc16
DATA_ROWS_START = 0  # данные начинаются с первого блока после header

BLOCKS_PER_REGION = GRID_COLS * GRID_ROWS - HEADER_BLOCKS
# Формат 3-х регионов: мы делим кадр на 3 горизонтальные полосы
# Каждая полоса — независимая копия тех же данных.
# При декодировании — majority vote по 3 копиям на каждый блок.

# Пересчитываем: сетка делится на 3 полосы
REGION_ROWS = GRID_ROWS // 3
BLOCKS_PER_REGION = GRID_COLS * REGION_ROWS - HEADER_BLOCKS  # данных на регион

# Гарантируем что хватает строк
assert REGION_ROWS >= 2, f"Слишком мало строк ({REGION_ROWS}) для региона"

# ═════════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ═════════════════════════════════════════════════════════════════════════════

def nibbles_from_bytes(data: bytes):
    """Генератор: байты → 4-битные значения (0-15)."""
    for b in data:
        yield (b >> 4) & 0x0F
        yield b & 0x0F


def bytes_from_nibbles(nibs: list[int]) -> bytes:
    """Список 4-битных → байты. Если нечётное — дополняет нулём."""
    out = bytearray()
    for i in range(0, len(nibs) - 1, 2):
        out.append((nibs[i] << 4) | (nibs[i + 1] & 0x0F))
    if len(nibs) % 2 == 1:
        out.append((nibs[-1] << 4))
    return bytes(out)


def crc16(data: bytes) -> int:
    """CRC-16/CCITT."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def encode_uint16_as_nibbles(value: int) -> list[int]:
    """uint16 → 4 nibble (big-endian)."""
    return [
        (value >> 12) & 0xF,
        (value >> 8) & 0xF,
        (value >> 4) & 0xF,
        value & 0xF,
    ]


def decode_nibbles_to_uint16(nibs: list[int]) -> int:
    """4 nibble → uint16."""
    return (nibs[0] << 12) | (nibs[1] << 8) | (nibs[2] << 4) | nibs[3]


def nearest_palette_index(bgr_pixel, palette_np=PALETTE_NP) -> int:
    """Ближайший цвет палитры для BGR-пикселя (евклидово расстояние)."""
    diff = palette_np - np.array(bgr_pixel, dtype=np.int32)
    dists = np.sum(diff * diff, axis=1)
    return int(np.argmin(dists))


# ═════════════════════════════════════════════════════════════════════════════
#  ШИФРОВАНИЕ AES-256-GCM
# ═════════════════════════════════════════════════════════════════════════════

SALT_LEN = 16
NONCE_LEN = 12


def derive_key(password: str, salt: bytes) -> bytes:
    """PBKDF2 → 256-bit ключ."""
    if not HAS_CRYPTO:
        raise RuntimeError("Библиотека cryptography не установлена: pip install cryptography")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=salt, iterations=480_000)
    return kdf.derive(password.encode("utf-8"))


def encrypt_data(data: bytes, password: str) -> bytes:
    """Возвращает salt(16) + nonce(12) + ciphertext+tag."""
    salt = os.urandom(SALT_LEN)
    key = derive_key(password, salt)
    nonce = os.urandom(NONCE_LEN)
    ct = AESGCM(key).encrypt(nonce, data, None)
    return salt + nonce + ct


def decrypt_data(blob: bytes, password: str) -> bytes:
    """Обратно расшифровывает то, что вернул encrypt_data."""
    salt = blob[:SALT_LEN]
    nonce = blob[SALT_LEN:SALT_LEN + NONCE_LEN]
    ct = blob[SALT_LEN + NONCE_LEN:]
    key = derive_key(password, salt)
    return AESGCM(key).decrypt(nonce, ct, None)


# ═════════════════════════════════════════════════════════════════════════════
#  ФОРМАТ ДАННЫХ (бинарный заголовок)
# ═════════════════════════════════════════════════════════════════════════════
# Заголовок перед данными файла (всё little-endian):
#
#   magic        4 bytes  b"YTC1"
#   flags        1 byte   bit0=encrypted
#   filename_len 2 bytes  uint16
#   filename     N bytes  UTF-8
#   file_size    8 bytes  uint64 (оригинальный размер до шифрования)
#   payload_size 8 bytes  uint64 (точный размер payload в байтах)
#   sha256       32 bytes хеш оригинальных данных
#   payload      ...      данные файла (или зашифрованные)
#
MAGIC = b"YTC1"


def build_header(filename: str, file_size: int, payload_size: int,
                 sha256_hash: bytes, encrypted: bool) -> bytes:
    """Собирает бинарный заголовок."""
    fname_bytes = filename.encode("utf-8")
    flags = 0x01 if encrypted else 0x00
    hdr = bytearray()
    hdr += MAGIC
    hdr += struct.pack("<B", flags)
    hdr += struct.pack("<H", len(fname_bytes))
    hdr += fname_bytes
    hdr += struct.pack("<Q", file_size)
    hdr += struct.pack("<Q", payload_size)
    hdr += sha256_hash
    return bytes(hdr)


def parse_header(data: bytes):
    """
    Разбирает заголовок. Возвращает (filename, file_size, payload_size, sha256, encrypted, header_len)
    или None если magic не совпал.
    """
    if len(data) < 4 or data[:4] != MAGIC:
        return None
    pos = 4
    flags = data[pos]; pos += 1
    fname_len = struct.unpack("<H", data[pos:pos+2])[0]; pos += 2
    filename = data[pos:pos+fname_len].decode("utf-8"); pos += fname_len
    file_size = struct.unpack("<Q", data[pos:pos+8])[0]; pos += 8
    payload_size = struct.unpack("<Q", data[pos:pos+8])[0]; pos += 8
    sha256 = data[pos:pos+32]; pos += 32
    encrypted = bool(flags & 0x01)
    return filename, file_size, payload_size, sha256, encrypted, pos


# ═════════════════════════════════════════════════════════════════════════════
#  ENCODER
# ═════════════════════════════════════════════════════════════════════════════

class Encoder:
    """Кодирует файл в видео с 3x-redundancy, frame numbering, CRC, AES."""

    def __init__(self, password: str | None = None):
        self.password = password

    def _make_frame(self, frame_num: int, data_nibbles: list[int]) -> np.ndarray:
        """
        Рисует один кадр: чёрный фон, затем 3 полосы-региона
        с одинаковыми данными. В начале каждого региона — 8 служебных блоков
        (4 — номер кадра uint16, 4 — CRC16 данных кадра).
        """
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # CRC данных кадра
        data_bytes_for_crc = bytes_from_nibbles(data_nibbles)
        crc = crc16(data_bytes_for_crc)

        frame_nibs = encode_uint16_as_nibbles(frame_num)
        crc_nibs = encode_uint16_as_nibbles(crc)
        header_nibs = frame_nibs + crc_nibs  # 8 nibble

        full_nibs = header_nibs + data_nibbles

        # Рисуем 3 региона (горизонтальные полосы)
        for region in range(3):
            row_offset = region * REGION_ROWS
            for idx, nib in enumerate(full_nibs):
                row = row_offset + idx // GRID_COLS
                col = idx % GRID_COLS
                if row >= row_offset + REGION_ROWS:
                    break
                x1 = col * STEP_X
                y1 = row * STEP_Y
                x2 = x1 + BLOCK_W
                y2 = y1 + BLOCK_H
                if x2 <= WIDTH and y2 <= HEIGHT:
                    frame[y1:y2, x1:x2] = PALETTE_BGR[nib]

        return frame

    def encode(self, input_file: str, output_file: str) -> bool:
        """Основной пайплайн кодирования."""
        # 1. Читаем файл
        with open(input_file, "rb") as f:
            raw_data = f.read()

        file_size = len(raw_data)
        sha256 = hashlib.sha256(raw_data).digest()
        filename = os.path.basename(input_file)
        encrypted = self.password is not None

        print(f"  Файл: {filename}")
        print(f"  Размер: {file_size:,} байт ({file_size/1024/1024:.2f} MB)")
        print(f"  SHA-256: {sha256.hex()[:16]}...")
        print(f"  Шифрование: {'AES-256-GCM' if encrypted else 'нет'}")

        # 2. Шифруем
        if encrypted:
            payload = encrypt_data(raw_data, self.password)
            print(f"  Зашифровано: {len(payload):,} байт")
        else:
            payload = raw_data

        # 3. Собираем заголовок + payload
        header = build_header(filename, file_size, len(payload), sha256, encrypted)
        full_data = header + payload

        # 4. Конвертируем в nibbles
        all_nibbles = list(nibbles_from_bytes(full_data))
        total_nibbles = len(all_nibbles)

        # 5. Разбиваем на кадры
        nibs_per_frame = BLOCKS_PER_REGION
        total_frames = math.ceil(total_nibbles / nibs_per_frame)

        print(f"  Блоков данных/кадр: {nibs_per_frame}")
        print(f"  Всего кадров: {total_frames}")
        print(f"  Длительность: {total_frames/FPS:.1f} сек")

        # 6. Создаём временную папку для кадров
        temp_dir = tempfile.mkdtemp(prefix="ytcloud_")

        try:
            # 7. Генерируем кадры
            for fnum in tqdm(range(total_frames), desc="  Кадры", unit="кадр"):
                start = fnum * nibs_per_frame
                end = min(start + nibs_per_frame, total_nibbles)
                chunk = all_nibbles[start:end]
                # Дополняем нулями если последний кадр неполный
                if len(chunk) < nibs_per_frame:
                    chunk = chunk + [0] * (nibs_per_frame - len(chunk))

                frame = self._make_frame(fnum, chunk)
                path = os.path.join(temp_dir, f"frame_{fnum:06d}.png")
                cv2.imwrite(path, frame)

            # 8. Собираем MP4 через FFmpeg
            print("  Сборка MP4...")
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(FPS),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                # Настройки для максимального выживания через YouTube:
                # CRF 0 = lossless (YouTube всё равно пережмёт, но мы даём
                # максимально чистый исходник).
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "0",
                "-pix_fmt", "yuv444p",
                "-an",
                "-movflags", "+faststart",
                output_file,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                # Fallback: yuv420p (некоторые ffmpeg не поддерживают yuv444p)
                cmd[cmd.index("yuv444p")] = "yuv420p"
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"  ОШИБКА ffmpeg: {r.stderr[-500:]}")
                    return False

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        out_size = os.path.getsize(output_file)
        print(f"  Видео: {output_file}")
        print(f"  Размер видео: {out_size:,} байт ({out_size/1024/1024:.2f} MB)")
        print("  Готово!")
        return True


# ═════════════════════════════════════════════════════════════════════════════
#  DECODER
# ═════════════════════════════════════════════════════════════════════════════

class Decoder:
    """Декодирует видео обратно в файл с majority-voting по 3 регионам."""

    def __init__(self, password: str | None = None):
        self.password = password
        # Кеш nearest-color
        self._cache: dict[tuple, int] = {}

    def _pixel_to_index(self, bgr) -> int:
        """BGR → индекс палитры (с кешем)."""
        key = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        if key in self._cache:
            return self._cache[key]
        idx = nearest_palette_index(key)
        self._cache[key] = idx
        return idx

    def _sample_block(self, frame: np.ndarray, row: int, col: int) -> int:
        """
        Считывает один блок, сэмплируя центральную область (60% площади)
        и возвращая медианный цвет → индекс палитры.
        """
        x1 = col * STEP_X
        y1 = row * STEP_Y
        # Центральная 60% область
        margin_x = BLOCK_W // 5
        margin_y = BLOCK_H // 5
        cx1 = x1 + margin_x
        cy1 = y1 + margin_y
        cx2 = x1 + BLOCK_W - margin_x
        cy2 = y1 + BLOCK_H - margin_y

        if cx2 > frame.shape[1] or cy2 > frame.shape[0]:
            return 0

        region = frame[cy1:cy2, cx1:cx2]
        # Медиана по каждому каналу — устойчива к шуму сжатия
        median_bgr = np.median(region.reshape(-1, 3), axis=0).astype(np.int32)
        return self._pixel_to_index(median_bgr)

    def _decode_region(self, frame: np.ndarray, region_idx: int) -> list[int]:
        """Декодирует один регион кадра → список nibble (header + data)."""
        row_offset = region_idx * REGION_ROWS
        nibs = []
        total_blocks = HEADER_BLOCKS + BLOCKS_PER_REGION
        for idx in range(total_blocks):
            row = row_offset + idx // GRID_COLS
            col = idx % GRID_COLS
            if row >= row_offset + REGION_ROWS:
                break
            nibs.append(self._sample_block(frame, row, col))
        return nibs

    def _majority_vote(self, regions: list[list[int]]) -> list[int]:
        """Majority voting по 3 регионам, поблочно."""
        max_len = max(len(r) for r in regions)
        result = []
        for i in range(max_len):
            votes = []
            for r in regions:
                if i < len(r):
                    votes.append(r[i])
            if not votes:
                result.append(0)
                continue
            # Считаем голоса
            counts: dict[int, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            # Побеждает большинство
            winner = max(counts, key=counts.get)
            result.append(winner)
        return result

    def decode(self, video_file: str, output_dir: str = ".") -> bool:
        """Основной пайплайн декодирования."""

        if not os.path.exists(video_file):
            print(f"  ОШИБКА: файл не найден: {video_file}")
            return False

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("  ОШИБКА: не удалось открыть видео")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Кадров: {total_frames}, FPS: {vid_fps:.1f}, {vid_w}x{vid_h}")

        # Собираем данные кадр за кадром
        frame_data: dict[int, list[int]] = {}  # frame_num → data nibbles
        errors = 0

        for _ in tqdm(range(total_frames), desc="  Декодирование", unit="кадр"):
            ret, raw_frame = cap.read()
            if not ret:
                break

            # Масштабируем к 1920x1080 если нужно
            if raw_frame.shape[1] != WIDTH or raw_frame.shape[0] != HEIGHT:
                raw_frame = cv2.resize(raw_frame, (WIDTH, HEIGHT),
                                       interpolation=cv2.INTER_AREA)

            # Декодируем 3 региона
            regions = [self._decode_region(raw_frame, r) for r in range(3)]

            # Majority vote
            voted = self._majority_vote(regions)

            if len(voted) < HEADER_BLOCKS:
                errors += 1
                continue

            # Извлекаем header: frame_num (4 nib) + crc16 (4 nib)
            frame_num = decode_nibbles_to_uint16(voted[:4])
            frame_crc = decode_nibbles_to_uint16(voted[4:8])
            data_nibs = voted[8:]

            # Проверяем CRC
            data_bytes_check = bytes_from_nibbles(data_nibs)
            actual_crc = crc16(data_bytes_check)

            if actual_crc != frame_crc:
                # CRC не совпал — возможно повреждён кадр, но всё равно сохраняем
                errors += 1

            # Дедупликация: если кадр с таким номером уже есть, берём тот что с верным CRC
            if frame_num not in frame_data or actual_crc == frame_crc:
                frame_data[frame_num] = data_nibs

        cap.release()

        print(f"  Уникальных кадров: {len(frame_data)}")
        if errors:
            print(f"  CRC-ошибок: {errors}")

        if not frame_data:
            print("  ОШИБКА: не удалось декодировать ни одного кадра")
            return False

        # Собираем nibbles в правильном порядке
        max_frame = max(frame_data.keys())
        all_nibbles: list[int] = []
        missing_frames = []
        for fnum in range(max_frame + 1):
            if fnum in frame_data:
                all_nibbles.extend(frame_data[fnum])
            else:
                missing_frames.append(fnum)
                # Заполняем нулями — данные потеряны
                all_nibbles.extend([0] * BLOCKS_PER_REGION)

        if missing_frames:
            print(f"  ВНИМАНИЕ: пропущены кадры: {missing_frames[:20]}{'...' if len(missing_frames) > 20 else ''}")

        # Конвертируем nibbles → bytes
        raw_bytes = bytes_from_nibbles(all_nibbles)

        # Парсим заголовок
        parsed = parse_header(raw_bytes)
        if parsed is None:
            print("  ОШИБКА: заголовок не найден (magic YTC1 не совпал)")
            # Сохраняем сырые данные
            fallback = os.path.join(output_dir, "decoded_raw.bin")
            os.makedirs(output_dir, exist_ok=True)
            with open(fallback, "wb") as f:
                f.write(raw_bytes)
            print(f"  Сырые данные сохранены: {fallback}")
            return False

        filename, file_size, payload_size, expected_sha, encrypted, hdr_len = parsed
        print(f"  Файл: {filename}")
        print(f"  Размер оригинала: {file_size:,} байт")
        print(f"  Размер payload: {payload_size:,} байт")
        print(f"  Шифрование: {'AES-256-GCM' if encrypted else 'нет'}")

        # Извлекаем payload — точно payload_size байт
        payload = raw_bytes[hdr_len:hdr_len + payload_size]

        # Расшифровываем если нужно
        if encrypted:
            if not self.password:
                print("  ОШИБКА: файл зашифрован, но пароль не указан")
                return False
            if not HAS_CRYPTO:
                print("  ОШИБКА: нужна библиотека cryptography: pip install cryptography")
                return False
            try:
                file_data = decrypt_data(payload, self.password)
            except Exception as e:
                print(f"  ОШИБКА расшифровки: {e}")
                return False
        else:
            file_data = payload[:file_size]

        # Обрезаем до оригинального размера
        file_data = file_data[:file_size]

        # Проверяем SHA-256
        actual_sha = hashlib.sha256(file_data).digest()
        if actual_sha == expected_sha:
            print(f"  SHA-256: совпадает!")
        else:
            print(f"  ВНИМАНИЕ: SHA-256 НЕ совпадает!")
            print(f"    Ожидалось: {expected_sha.hex()[:16]}...")
            print(f"    Получено:  {actual_sha.hex()[:16]}...")

        # Сохраняем файл
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        counter = 1
        base, ext = os.path.splitext(filename)
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{base}_{counter}{ext}")
            counter += 1

        with open(output_path, "wb") as f:
            f.write(file_data)

        print(f"  Сохранено: {output_path}")
        print(f"  Размер: {len(file_data):,} байт")

        if len(file_data) == file_size and actual_sha == expected_sha:
            print("  Файл восстановлен без потерь!")
            return True
        else:
            print("  Файл восстановлен с возможными ошибками")
            return False


# ═════════════════════════════════════════════════════════════════════════════
#  VERIFY — roundtrip-тест
# ═════════════════════════════════════════════════════════════════════════════

def verify(input_file: str, password: str | None = None) -> bool:
    """Encode → Decode → сравнить хеши."""
    import time

    print("\n  Roundtrip-тест")
    print("  " + "=" * 50)

    temp_dir = tempfile.mkdtemp(prefix="ytcloud_verify_")
    video_path = os.path.join(temp_dir, "test.mp4")
    decode_dir = os.path.join(temp_dir, "decoded")
    os.makedirs(decode_dir)

    try:
        # Encode
        t0 = time.time()
        enc = Encoder(password)
        if not enc.encode(input_file, video_path):
            print("  Кодирование провалилось")
            return False
        t_enc = time.time() - t0

        # Decode
        t0 = time.time()
        dec = Decoder(password)
        if not dec.decode(video_path, decode_dir):
            print("  Декодирование провалилось")
            return False
        t_dec = time.time() - t0

        # Сравниваем
        original = open(input_file, "rb").read()
        orig_hash = hashlib.sha256(original).hexdigest()

        decoded_file = os.path.join(decode_dir, os.path.basename(input_file))
        if not os.path.exists(decoded_file):
            # Ищем любой файл
            files = os.listdir(decode_dir)
            if files:
                decoded_file = os.path.join(decode_dir, files[0])
            else:
                print("  Декодированный файл не найден")
                return False

        decoded = open(decoded_file, "rb").read()
        dec_hash = hashlib.sha256(decoded).hexdigest()

        print(f"\n  Оригинал:      {len(original):,} байт, SHA-256: {orig_hash[:16]}...")
        print(f"  Декодированный: {len(decoded):,} байт, SHA-256: {dec_hash[:16]}...")
        print(f"  Кодирование: {t_enc:.1f} сек")
        print(f"  Декодирование: {t_dec:.1f} сек")

        if orig_hash == dec_hash:
            print("  ТЕСТ ПРОЙДЕН — данные идентичны!")
            return True
        else:
            print("  ТЕСТ НЕ ПРОЙДЕН — данные различаются!")
            # Найти первое различие
            for i in range(min(len(original), len(decoded))):
                if original[i] != decoded[i]:
                    print(f"  Первое различие на байте {i}")
                    break
            return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def read_password(key_file: str = "key.txt") -> str | None:
    """Читает пароль из key.txt если он есть."""
    if os.path.exists(key_file):
        with open(key_file, "r", encoding="utf-8") as f:
            pw = f.read().strip()
        if pw:
            print(f"  Ключ загружен из {key_file}")
            return pw
    return None


def main():
    parser = argparse.ArgumentParser(
        description="yt-cloud — стеганографическое хранение файлов в видео",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Примеры:
  %(prog)s encode secret.zip               → output.mp4
  %(prog)s encode secret.zip video.mp4     → video.mp4
  %(prog)s decode video.mp4                → текущая папка
  %(prog)s decode video.mp4 ./recovered/   → ./recovered/
  %(prog)s verify secret.zip               → encode→decode→сравнить

Шифрование:
  Создайте файл key.txt с паролем, или используйте --key:
  %(prog)s encode --key mypassword secret.zip
""",
    )

    parser.add_argument("command", choices=["encode", "decode", "verify"],
                        help="encode|decode|verify")
    parser.add_argument("input", help="Входной файл (encode/verify) или видео (decode)")
    parser.add_argument("output", nargs="?", default=None,
                        help="Выходной файл/папка (по умолчанию: output.mp4 / .)")
    parser.add_argument("--key", "-k", default=None,
                        help="Пароль для шифрования (или key.txt)")

    args = parser.parse_args()

    # Определяем пароль
    password = args.key or read_password()

    print()
    print("=" * 58)
    print("  yt-cloud coder v2.0")
    print("=" * 58)

    if args.command == "encode":
        if not os.path.exists(args.input):
            print(f"  ОШИБКА: файл не найден: {args.input}")
            sys.exit(1)
        output = args.output or "output.mp4"
        enc = Encoder(password)
        ok = enc.encode(args.input, output)
        sys.exit(0 if ok else 1)

    elif args.command == "decode":
        output_dir = args.output or "."
        dec = Decoder(password)
        ok = dec.decode(args.input, output_dir)
        sys.exit(0 if ok else 1)

    elif args.command == "verify":
        if not os.path.exists(args.input):
            print(f"  ОШИБКА: файл не найден: {args.input}")
            sys.exit(1)
        ok = verify(args.input, password)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
