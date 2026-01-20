#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDITONES Score Converter | MIDITONES 樂譜轉換器
================================================

Convert MIDITONES-generated score[] bytestream (.c) into a simple Note array (Hz + frames).
把 MIDITONES 產生的 score[] bytestream (.c) 轉成簡單的 Note 陣列（Hz + 幀數）

Bytestream Format (based on LenShustek/miditones README):
依據 LenShustek/miditones README 的 bytestream 格式：
  - 'Pt' header: contains length, flags, and number of tone generators used
    'Pt' 標頭：含長度、旗標、使用的 tone generators 數
  - Commands 指令: 9t nn [vv] (note on), 8t (note off), Ct ii (program change), F0/E0 (end)
  - Delay 延遲: two bytes, 15-bit big-endian milliseconds 兩個 byte，15-bit 大端毫秒

Usage 使用方式:
  - Default: looks for score.c in current directory, outputs song.c
    預設：尋找當前目錄的 score.c，輸出 song.c
  - Use -t=1 to generate mono stream; for >1 tone generators, use --tone to select
    使用 -t=1 產生單音流；若 >1，可用 --tone 指定抽取哪個 tone generator

Author: Your Name | 作者：你的名字
License: MIT
"""

import argparse
import os
import re
import glob
import math
import sys
from typing import List, Tuple, Optional

# Default frame rate for output | 預設輸出幀率
FRAMES_PER_SEC_DEFAULT = 60.0

# Note names for display | 音符名稱供顯示用
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_note_to_hz(n: int) -> int:
    """
    Convert MIDI note number to frequency in Hz.
    將 MIDI 音符編號轉換為頻率（Hz）。
    
    Args:
        n: MIDI note number (0-127) | MIDI 音符編號（0-127）
    
    Returns:
        Frequency in Hz | 頻率（Hz）
    """
    return int(round(440.0 * (2.0 ** ((n - 69) / 12.0))))


def midi_note_to_name(n: int) -> str:
    """
    Convert MIDI note number to note name (e.g., C4, A#3).
    將 MIDI 音符編號轉換為音符名稱（如 C4、A#3）。
    
    Args:
        n: MIDI note number (0-127) | MIDI 音符編號（0-127）
    
    Returns:
        Note name string | 音符名稱字串
    """
    octave = (n // 12) - 1
    name = NOTE_NAMES[n % 12]
    return f"{name}{octave}"


def strip_c_comments(s: str) -> str:
    """
    Remove C-style comments (// and /* */).
    移除 C 風格註解（// 和 /* */）。
    
    Args:
        s: Input string with C code | 輸入的 C 程式碼字串
    
    Returns:
        String with comments removed | 移除註解後的字串
    """
    s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    return s


def find_score_initializer(text: str) -> str:
    """
    Find the content of score[] = { ... } initializer.
    找到 score[] = { ... } 的內容（含可能的 PROGMEM 等修飾）。
    
    Args:
        text: C source code | C 原始碼
    
    Returns:
        Content between { and } | 大括號之間的內容
    
    Raises:
        RuntimeError: If initializer not found | 找不到初始化區塊時
    """
    m = re.search(r'score\s*\[\s*\]\s*=\s*\{', text)
    if not m:
        # Fallback: find the first { ... }, but less safe
        # 若找不到，退而求其次找第一個 { ... }，但這較不安全
        m = re.search(r'\{', text)
        if not m:
            raise RuntimeError("Cannot find '{' for score[] initializer | 找不到 score[] 初始化的 '{'")
    start = m.end()  # Points to after '{' | 指向 '{' 之後
    # Find matching '}' using simple stack matching
    # 從 start 起找對應的結束 '}'（簡單堆疊匹配）
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i
                return text[start:end]
        i += 1
    raise RuntimeError("Cannot find matching '}' for score[] initializer | 找不到 score[] 初始化的對應 '}'")


def parse_c_initializer_bytes(init_text: str) -> List[int]:
    """
    Parse C array initializer bytes. Supports decimal, hex (0x??), and char literals ('P').
    解析 C 陣列初始化器的 byte。支援十進位、十六進位 0x??、字元常值 'P' 等。
    
    Args:
        init_text: Content between { and } | 大括號之間的內容
    
    Returns:
        List of byte values (0-255) | byte 值列表（0-255）
    """
    tokens = [tok.strip() for tok in init_text.split(',')]
    bytes_out: List[int] = []
    import ast
    for tok in tokens:
        if not tok:
            continue
        # Comments already removed | 註解已在之前移除
        # Character literal | 字元常值
        if re.fullmatch(r"'(\\.|[^'])'", tok):
            try:
                ch = ast.literal_eval(tok)  # Convert to Python string | 轉成 Python 字串
                if isinstance(ch, str) and len(ch) == 1:
                    bytes_out.append(ord(ch))
                    continue
            except Exception:
                pass
            raise ValueError(f"Cannot parse char literal | 無法解析字元常值: {tok}")
        # Hexadecimal | 十六進位
        if tok.lower().startswith('0x'):
            try:
                val = int(tok, 16)
            except Exception:
                raise ValueError(f"Cannot parse hex value | 無法解析十六進位數值: {tok}")
            if not (0 <= val <= 255):
                raise ValueError(f"Value out of range 0..255 | 數值超出 0..255: {tok}")
            bytes_out.append(val)
            continue
        # Decimal | 十進位
        try:
            val = int(tok, 10)
        except Exception:
            # Some compiler outputs may have casts or macros, try stripping suffixes
            # 有些編譯器輸出可能帶有 cast 或巨集，嘗試剔除尾碼
            tok2 = re.sub(r'[uUlLfF]+$', '', tok)
            val = int(tok2, 10)
        if not (0 <= val <= 255):
            raise ValueError(f"Value out of range 0..255 | 數值超出 0..255: {tok}")
        bytes_out.append(val)
    return bytes_out


class MiditonesHeader:
    """
    Represents the 'Pt' header in MIDITONES bytestream.
    表示 MIDITONES bytestream 中的 'Pt' 標頭。
    """
    def __init__(self):
        self.present = False        # Whether header exists | 標頭是否存在
        self.length = 0             # Header length in bytes | 標頭長度（bytes）
        self.flags1 = 0             # First flags byte | 第一個旗標 byte
        self.flags2 = 0             # Second flags byte | 第二個旗標 byte
        self.tone_generators = 0    # Number of tone generators | tone generators 數量


def parse_miditones_bytestream(all_bytes: List[int], selected_tone: int = 0):
    """
    Parse MIDITONES bytestream and extract note segments.
    解析 MIDITONES bytestream 並抽取音符段落。
    
    Args:
        all_bytes: Raw bytes from score[] | score[] 的原始 bytes
        selected_tone: Tone generator to extract (default: 0) | 要抽取的 tone generator（預設 0）
    
    Returns:
        Tuple of:
            segments: List[Tuple[start_ms, end_ms, freq_hz, midi_note_or_None]]
            header: MiditonesHeader
    """
    i = 0
    header = MiditonesHeader()
    
    # Check for 'Pt' header | 檢查 'Pt' 標頭
    if len(all_bytes) >= 6 and all_bytes[0] == ord('P') and all_bytes[1] == ord('t'):
        header.present = True
        hdr_len = all_bytes[2]
        if hdr_len < 6 or hdr_len > len(all_bytes):
            raise RuntimeError(f"Invalid header length | 不合理的標頭長度: {hdr_len}")
        header.length = hdr_len
        if hdr_len >= 6:
            header.flags1 = all_bytes[3]
            header.flags2 = all_bytes[4]
            header.tone_generators = all_bytes[5]
        # Skip remaining header bytes (if length > 6)
        # 其餘標頭位元（如果 length > 6）忽略
        i = hdr_len

    vol_present = bool(header.flags1 & 0x80) if header.present else False

    t_ms = 0  # Current time in ms | 目前時間（毫秒）
    cur_freq = 0
    cur_midi: Optional[int] = None
    last_change_ms = 0

    segments: List[Tuple[int,int,int,Optional[int]]] = []

    def close_segment(now_ms: int):
        """Close current segment and record it. | 關閉當前段落並記錄。"""
        nonlocal last_change_ms, cur_freq, cur_midi
        if now_ms > last_change_ms:
            segments.append((last_change_ms, now_ms, cur_freq, cur_midi))
        last_change_ms = now_ms

    # Parse byte by byte | 逐 byte 解析
    while i < len(all_bytes):
        b = all_bytes[i]
        if b & 0x80:  # Command | 指令
            # Note on: 0x90..0x9F
            if 0x90 <= b <= 0x9F:
                if i + 1 >= len(all_bytes):
                    raise RuntimeError("Bytestream ended at note-on | bytestream 結尾於 note-on")
                tone = b & 0x0F
                midi_note = all_bytes[i+1]
                i += 2
                if vol_present:
                    if i >= len(all_bytes):
                        raise RuntimeError("Bytestream ended at velocity | bytestream 結尾於 velocity")
                    velocity = all_bytes[i]
                    i += 1  # Parsed but not used | 解析但不使用
                if tone == selected_tone:
                    # Per spec: note-on replaces the previous note
                    # 依規格：note-on 會「取代」前一個音
                    if cur_freq != 0 or cur_midi is not None:
                        close_segment(t_ms)
                    cur_midi = midi_note
                    cur_freq = midi_note_to_hz(midi_note)
                # Ignore other tone generators | 其他 tone generator 忽略
                continue
            # Note off: 0x80..0x8F
            if 0x80 <= b <= 0x8F:
                tone = b & 0x0F
                i += 1
                if tone == selected_tone:
                    if cur_freq != 0 or cur_midi is not None:
                        close_segment(t_ms)
                        cur_freq = 0
                        cur_midi = None
                    else:
                        # Already silent, no action | 本就靜音，不動作
                        pass
                continue
            # Program change (instrument): 0xC0..0xCF
            if 0xC0 <= b <= 0xCF:
                # Next byte is instrument | 下一 byte 為 instrument
                i += 2
                continue
            # End of score | 樂譜結束
            if b == 0xF0 or b == 0xE0:
                i += 1
                # E0 (restart) also treated as end here | E0（restart）在這裡也視為結束
                break
            # Other unimplemented commands: skip 1 byte to avoid infinite loop
            # 其他未實作命令：略過一個 byte，避免死循環
            i += 1
        else:
            # Delay: two bytes, 15-bit big-endian milliseconds
            # 延遲：兩個 byte，15-bit 大端毫秒
            if i + 1 >= len(all_bytes):
                raise RuntimeError("Bytestream ended at delay high byte | bytestream 結尾於延遲的上位元")
            b0 = b & 0x7F
            b1 = all_bytes[i+1]
            delay_ms = (b0 << 8) | b1
            t_ms += delay_ms
            i += 2

    # Close the last segment | 關閉最後一段
    if cur_freq != 0 or cur_midi is not None:
        close_segment(t_ms)
    else:
        # If the last segment is a rest with duration, add it too
        # 若最後一段是休止符且有持續時間，也補上
        if t_ms > last_change_ms:
            segments.append((last_change_ms, t_ms, 0, None))

    return segments, header


def quantize_segments_to_notes(segments: List[Tuple[int,int,int,Optional[int]]],
                               fps: float) -> List[Tuple[int,int,Optional[int]]]:
    """
    Quantize time segments to frame-based notes.
    將時間段落量化成幀數音符。
    
    Converts (start_ms, end_ms, freq_hz, midi_opt) to (freq_hz, frames, midi_opt).
    Merges adjacent segments with the same frequency; minimum 1 frame per note.
    把 (start_ms, end_ms, freq_hz, midi_opt) 轉成 (freq_hz, frames, midi_opt)。
    會合併相鄰相同頻率的段落；每段至少 1 幀。
    
    Args:
        segments: List of time segments | 時間段落列表
        fps: Target frame rate | 目標幀率
    
    Returns:
        List of (frequency, frames, midi_note_or_None) | (頻率, 幀數, midi音符或None) 列表
    """
    # Merge adjacent segments with same frequency | 合併相鄰相同頻率
    merged: List[Tuple[int,int,int,Optional[int]]] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            ls, le, lf, lm = merged[-1]
            s, e, f, m = seg
            if lf == f and ls <= s <= le:  # Continuous and same frequency | 連續且同頻率
                merged[-1] = (ls, e, lf, lm if lm is not None else m)
            else:
                merged.append(seg)

    frame_ms = 1000.0 / fps
    notes: List[Tuple[int,int,Optional[int]]] = []
    error_carry = 0.0  # Accumulate error for more accurate total duration | 誤差累積讓總時長更準

    for (s, e, f, m) in merged:
        dur_ms = max(0.0, float(e - s))
        # Add accumulated error to current segment | 把誤差累積到當前段落
        adj_ms = dur_ms + error_carry
        frames = int(round(adj_ms / frame_ms))
        if frames <= 0 and dur_ms > 0.0:
            frames = 1
        # Update error: difference between actual output and expected duration
        # 更新誤差：實際輸出 frames * frame_ms 與 dur_ms 的差
        error_carry = adj_ms - frames * frame_ms
        notes.append((f, frames, m))
    
    # Remove segments with 0 frames | 移除 frames=0 的段落
    notes = [(f, fr, m) for (f, fr, m) in notes if fr > 0]
    return notes


def choose_input_file(path_arg: Optional[str]) -> str:
    """
    Choose input file automatically or from argument.
    自動選擇輸入檔案或從參數取得。
    
    Priority | 優先順序:
    1. Specified path | 指定的路徑
    2. score.c in current directory | 當前目錄的 score.c
    3. Any .c file containing "score[] =" | 任何包含 "score[] =" 的 .c 檔
    4. The only .c file in directory | 目錄中唯一的 .c 檔
    
    Args:
        path_arg: User-specified path or None | 使用者指定的路徑或 None
    
    Returns:
        Path to input file | 輸入檔案路徑
    
    Raises:
        FileNotFoundError: If specified file doesn't exist | 指定檔案不存在時
        RuntimeError: If no suitable file found | 找不到合適檔案時
    """
    if path_arg:
        if not os.path.isfile(path_arg):
            raise FileNotFoundError(path_arg)
        return path_arg
    # 1) Try score.c | 嘗試 score.c
    if os.path.isfile("score.c"):
        return "score.c"
    # 2) Find .c files containing "score[] =" | 尋找包含 "score[] =" 的 .c 檔
    candidates = []
    for p in glob.glob("*.c"):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                if re.search(r'\bscore\s*\[\s*\]\s*=', f.read()):
                    candidates.append(p)
        except Exception:
            pass
    if candidates:
        # If multiple, choose the first one | 若有多個，選第一個
        return candidates[0]
    # 3) Fallback: the only .c file in directory | 退而求其次：目錄中唯一 .c
    cs = glob.glob("*.c")
    if len(cs) == 1:
        return cs[0]
    raise RuntimeError(
        "Cannot find input file. Use -i to specify, or name the file score.c | "
        "找不到輸入檔，請用 -i 指定，或把檔案命名為 score.c"
    )


def main():
    """Main entry point. | 主程式入口。"""
    ap = argparse.ArgumentParser(
        description="Convert MIDITONES score[] bytestream to Note array (Hz + frames) | "
                    "將 MIDITONES 的 score[] bytestream 轉成 Note 陣列（Hz + 幀數）"
    )
    ap.add_argument("-i", "--input",
                    help="Input .c file (default: auto-find score.c) | 輸入 .c 檔（預設自動尋找 score.c）")
    ap.add_argument("-o", "--output", default="song.c",
                    help="Output .c filename (default: song.c) | 輸出 .c 檔名（預設 song.c）")
    ap.add_argument("--tone", type=int, default=0,
                    help="Tone generator to extract (default: 0) | 選擇要抽取的 tone generator（預設 0）")
    ap.add_argument("--fps", type=float, default=FRAMES_PER_SEC_DEFAULT,
                    help="Output frame rate (default: 60) | 輸出幀率（預設 60）")
    args = ap.parse_args()

    in_path = choose_input_file(args.input)

    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    no_comments = strip_c_comments(text)
    init = find_score_initializer(no_comments)
    all_bytes = parse_c_initializer_bytes(init)
    segments, header = parse_miditones_bytestream(all_bytes, selected_tone=args.tone)

    if header.present:
        msg = (f"Detected 'Pt' header | 偵測到 'Pt' 標頭: "
               f"length={header.length}, flags1=0x{header.flags1:02X}, "
               f"flags2=0x{header.flags2:02X}, tone_generators={header.tone_generators}")
        print(msg)
        if header.tone_generators > 1:
            print(f"Warning | 警告: This bytestream requires {header.tone_generators} tone generators; "
                  f"only outputting mono version for tone {args.tone}. | "
                  f"這個 bytestream 需要 {header.tone_generators} 個 tone generators；"
                  f"目前只輸出 tone {args.tone} 的單音版本。", file=sys.stderr)

    notes = quantize_segments_to_notes(segments, fps=args.fps)

    # Generate output C file | 產出 C 檔
    with open(args.output, "w", encoding="utf-8") as out:
        out.write("// Auto-generated from MIDITONES bytestream | 由 MIDITONES bytestream 自動產生\n")
        out.write("// Units | 單位: frequency=Hz, duration=frames @ %.3f fps\n" % args.fps)
        out.write("// Source | 來源: %s\n\n" % os.path.basename(in_path))
        out.write("typedef struct { uint16_t frequency; uint16_t duration; } Note;\n\n")
        out.write("const Note song[] = {\n")
        for (freq, frames, midi_opt) in notes:
            if freq <= 0:
                out.write(f"    {{ 0, {frames} }},  // Rest | 休止符 ~{frames/args.fps:.3f}s\n")
            else:
                if midi_opt is not None:
                    name = midi_note_to_name(midi_opt)
                    out.write(f"    {{ {freq}, {frames} }},  // {name}, ~{frames/args.fps:.3f}s\n")
                else:
                    out.write(f"    {{ {freq}, {frames} }},  // ~{frames/args.fps:.3f}s\n")
        out.write("};\n")
        out.write("const unsigned song_len = sizeof(song)/sizeof(song[0]);\n")
    print(f"Generated {args.output} with {len(notes)} notes. | 已產生 {args.output}，共 {len(notes)} 筆 Note。")


if __name__ == "__main__":
    main()
