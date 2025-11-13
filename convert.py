#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 MIDITONES 產生的 score[] bytestream (.c) 轉成簡單的 Note 陣列（Hz + 60fps 幀數）
- 依據 LenShustek/miditones README 的 bytestream 格式：
  * 'Pt' header（含長度/旗標/使用的 tone generators 數）
  * 指令：9t nn [vv], 8t, Ct ii, F0, E0
  * 延遲：兩個 byte，15-bit big-endian 毫秒
- 預設尋找當前目錄的 score.c，輸出 song.c
- 預期你用 -t=1 產生單音流；如 >1，可用 --tone 指定抽取哪個 tone generator

作者：你
"""

import argparse
import os
import re
import glob
import math
import sys
from typing import List, Tuple, Optional

FRAMES_PER_SEC_DEFAULT = 60.0

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_note_to_hz(n: int) -> int:
    return int(round(440.0 * (2.0 ** ((n - 69) / 12.0))))

def midi_note_to_name(n: int) -> str:
    octave = (n // 12) - 1
    name = NOTE_NAMES[n % 12]
    return f"{name}{octave}"

def strip_c_comments(s: str) -> str:
    # 移除 // 和 /* */ 註解
    s = re.sub(r'//.*?$', '', s, flags=re.MULTILINE)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    return s

def find_score_initializer(text: str) -> str:
    # 找到 score[] = { ... } 的內容（含可能的 PROGMEM 等修飾）
    m = re.search(r'score\s*\[\s*\]\s*=\s*\{', text)
    if not m:
        # 若找不到，退而求其次找第一個 { ... }，但這較不安全
        m = re.search(r'\{', text)
        if not m:
            raise RuntimeError("找不到 score[] 初始化的 '{'")
    start = m.end()  # 指向 '{' 之後
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
    raise RuntimeError("找不到 score[] 初始化的對應 '}'")

def parse_c_initializer_bytes(init_text: str) -> List[int]:
    # 支援十進位、十六進位 0x??、字元常值 'P' 之類；忽略空白
    tokens = [tok.strip() for tok in init_text.split(',')]
    bytes_out: List[int] = []
    import ast
    for tok in tokens:
        if not tok:
            continue
        # 可能有多餘註解，但我們之前已移除
        # 字元常值
        if re.fullmatch(r"'(\\.|[^'])'", tok):
            try:
                ch = ast.literal_eval(tok)  # 轉成 Python 字串長度 1
                if isinstance(ch, str) and len(ch) == 1:
                    bytes_out.append(ord(ch))
                    continue
            except Exception:
                pass
            raise ValueError(f"無法解析字元常值: {tok}")
        # 十六進位
        if tok.lower().startswith('0x'):
            try:
                val = int(tok, 16)
            except Exception:
                raise ValueError(f"無法解析十六進位數值: {tok}")
            if not (0 <= val <= 255):
                raise ValueError(f"數值超出 0..255: {tok}")
            bytes_out.append(val)
            continue
        # 十進位
        try:
            val = int(tok, 10)
        except Exception:
            # 有些編譯器輸出可能帶有 cast 或巨集，這裡嘗試再剔除尾碼
            tok2 = re.sub(r'[uUlLfF]+$', '', tok)
            val = int(tok2, 10)
        if not (0 <= val <= 255):
            raise ValueError(f"數值超出 0..255: {tok}")
        bytes_out.append(val)
    return bytes_out

class MiditonesHeader:
    def __init__(self):
        self.present = False
        self.length = 0
        self.flags1 = 0
        self.flags2 = 0
        self.tone_generators = 0

def parse_miditones_bytestream(all_bytes: List[int], selected_tone: int = 0):
    """
    回傳：
      segments: List[Tuple[start_ms, end_ms, freq_hz, midi_note_or_None]]
      header: MiditonesHeader
    """
    i = 0
    header = MiditonesHeader()
    if len(all_bytes) >= 6 and all_bytes[0] == ord('P') and all_bytes[1] == ord('t'):
        header.present = True
        hdr_len = all_bytes[2]
        if hdr_len < 6 or hdr_len > len(all_bytes):
            raise RuntimeError(f"不合理的標頭長度: {hdr_len}")
        header.length = hdr_len
        if hdr_len >= 6:
            header.flags1 = all_bytes[3]
            header.flags2 = all_bytes[4]
            header.tone_generators = all_bytes[5]
        # 其餘標頭位元（如果 length > 6）忽略
        i = hdr_len

    vol_present = bool(header.flags1 & 0x80) if header.present else False

    t_ms = 0  # 目前時間（毫秒）
    cur_freq = 0
    cur_midi: Optional[int] = None
    last_change_ms = 0

    segments: List[Tuple[int,int,int,Optional[int]]] = []

    def close_segment(now_ms: int):
        nonlocal last_change_ms, cur_freq, cur_midi
        if now_ms > last_change_ms:
            segments.append((last_change_ms, now_ms, cur_freq, cur_midi))
        last_change_ms = now_ms

    # 逐 byte 解析
    while i < len(all_bytes):
        b = all_bytes[i]
        if b & 0x80:  # 指令
            # Note on: 0x90..0x9F
            if 0x90 <= b <= 0x9F:
                if i + 1 >= len(all_bytes):
                    raise RuntimeError("bytestream 結尾於 note-on")
                tone = b & 0x0F
                midi_note = all_bytes[i+1]
                i += 2
                if vol_present:
                    if i >= len(all_bytes):
                        raise RuntimeError("bytestream 結尾於 velocity")
                    velocity = all_bytes[i]
                    i += 1  # 我們解析但不使用
                if tone == selected_tone:
                    # 依規格：note-on 會「取代」前一個音
                    if cur_freq != 0 or cur_midi is not None:
                        close_segment(t_ms)
                    cur_midi = midi_note
                    cur_freq = midi_note_to_hz(midi_note)
                # 其他 tone generator 忽略，但不影響時間
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
                        # 本就靜音，不動作
                        pass
                continue
            # Program change (instrument): 0xC0..0xCF
            if 0xC0 <= b <= 0xCF:
                # 下一 byte 為 instrument
                i += 2
                continue
            # End of score
            if b == 0xF0 or b == 0xE0:
                i += 1
                # E0（restart）在這裡也視為結束
                break
            # 其他未實作命令：略過一個 byte 或視情況處理
            # 這裡保守做法：嘗試前進 1 byte，避免死循環
            i += 1
        else:
            # 延遲：兩個 byte，15-bit big-endian 毫秒
            if i + 1 >= len(all_bytes):
                raise RuntimeError("bytestream 結尾於延遲的上位元")
            b0 = b & 0x7F
            b1 = all_bytes[i+1]
            delay_ms = (b0 << 8) | b1
            t_ms += delay_ms
            i += 2

    # 關閉最後一段
    if cur_freq != 0 or cur_midi is not None:
        close_segment(t_ms)
    else:
        # 若最後一段是休止符，且有持續時間，也補上
        if t_ms > last_change_ms:
            segments.append((last_change_ms, t_ms, 0, None))

    return segments, header

def quantize_segments_to_notes(segments: List[Tuple[int,int,int,Optional[int]]],
                               fps: float) -> List[Tuple[int,int,Optional[int]]]:
    """
    將 (start_ms, end_ms, freq_hz, midi_opt) 量化成 (freq_hz, frames, midi_opt)
    會合併相鄰相同 freq 的段落；每段至少 1 幀
    """
    # 合併相鄰相同頻率
    merged: List[Tuple[int,int,int,Optional[int]]] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            ls, le, lf, lm = merged[-1]
            s, e, f, m = seg
            if lf == f and ls <= s <= le:  # 連續且同頻率
                merged[-1] = (ls, e, lf, lm if lm is not None else m)
            else:
                merged.append(seg)

    frame_ms = 1000.0 / fps
    notes: List[Tuple[int,int,Optional[int]]] = []
    error_carry = 0.0  # 以「誤差累積」讓總時長更準

    for (s, e, f, m) in merged:
        dur_ms = max(0.0, float(e - s))
        # 把誤差累積到當前段落
        adj_ms = dur_ms + error_carry
        frames = int(round(adj_ms / frame_ms))
        if frames <= 0 and dur_ms > 0.0:
            frames = 1
        # 更新誤差：實際輸出 frames * frame_ms 與 dur_ms 的差
        error_carry = adj_ms - frames * frame_ms
        notes.append((f, frames, m))
    # 移除 frames=0 的段落
    notes = [(f, fr, m) for (f, fr, m) in notes if fr > 0]
    return notes

def choose_input_file(path_arg: Optional[str]) -> str:
    if path_arg:
        if not os.path.isfile(path_arg):
            raise FileNotFoundError(path_arg)
        return path_arg
    # 1) score.c
    if os.path.isfile("score.c"):
        return "score.c"
    # 2) 尋找包含 "score[] =" 的 .c 檔
    candidates = []
    for p in glob.glob("*.c"):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                if re.search(r'\bscore\s*\[\s*\]\s*=', f.read()):
                    candidates.append(p)
        except Exception:
            pass
    if candidates:
        # 若有多個，選第一個
        return candidates[0]
    # 3) 退而求其次：目錄中唯一 .c
    cs = glob.glob("*.c")
    if len(cs) == 1:
        return cs[0]
    raise RuntimeError("找不到輸入檔，請用 -i 指定，或把檔案命名為 score.c")

def main():
    ap = argparse.ArgumentParser(description="將 MIDITONES 的 score[] bytestream .c 轉成 Note 陣列 song[]（Hz + 幀數）")
    ap.add_argument("-i", "--input", help="輸入 .c 檔（預設自動尋找 score.c）")
    ap.add_argument("-o", "--output", default="song.c", help="輸出 .c 檔名（預設 song.c）")
    ap.add_argument("--tone", type=int, default=0, help="選擇要抽取的 tone generator（預設 0）")
    ap.add_argument("--fps", type=float, default=FRAMES_PER_SEC_DEFAULT, help="轉出幀率（預設 60）")
    args = ap.parse_args()

    in_path = choose_input_file(args.input)

    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    no_comments = strip_c_comments(text)
    init = find_score_initializer(no_comments)
    all_bytes = parse_c_initializer_bytes(init)
    segments, header = parse_miditones_bytestream(all_bytes, selected_tone=args.tone)

    if header.present:
        msg = f"偵測到 'Pt' 標頭: 長度={header.length}, flags1=0x{header.flags1:02X}, flags2=0x{header.flags2:02X}, tone_generators={header.tone_generators}"
        print(msg)
        if header.tone_generators > 1:
            print(f"警告：這個 bytestream 需要 {header.tone_generators} 個 tone generators；目前只輸出 tone {args.tone} 的單音版本。", file=sys.stderr)

    notes = quantize_segments_to_notes(segments, fps=args.fps)

    # 產出 C 檔
    with open(args.output, "w", encoding="utf-8") as out:
        out.write("// Auto-generated from MIDITONES bytestream by miditones_score_to_notes.py\n")
        out.write("// 單位: frequency=Hz, duration=frames @ %.3f fps\n" % args.fps)
        out.write("// 來源: %s\n\n" % os.path.basename(in_path))
        out.write("typedef struct { uint16_t frequency; uint16_t duration; } Note;\n\n")
        out.write("const Note song[] = {\n")
        for (freq, frames, midi_opt) in notes:
            if freq <= 0:
                out.write(f"    {{ 0, {frames} }},  // 休止符 ~{frames/args.fps:.3f} 秒\n")
            else:
                if midi_opt is not None:
                    name = midi_note_to_name(midi_opt)
                    out.write(f"    {{ {freq}, {frames} }},  // {name}, ~{frames/args.fps:.3f} 秒\n")
                else:
                    out.write(f"    {{ {freq}, {frames} }},  // ~{frames/args.fps:.3f} 秒\n")
        out.write("};\n")
        out.write("const unsigned song_len = sizeof(song)/sizeof(song[0]);\n")
    print(f"已產生 {args.output}，共 {len(notes)} 筆 Note。")

if __name__ == "__main__":
    main()