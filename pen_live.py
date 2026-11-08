#!/usr/bin/env python3
"""
pen_live.py - write with the pen, see it as text.

Install:   pip install pygame numpy scikit-learn       (pyserial only for --serial)
Run:       python pen_live.py            (pen over WiFi, UDP 4210)
           python pen_live.py --ble      (pen over Bluetooth; pip install bleak)
           python pen_live.py --demo     (synthetic writing; train in pen_trainer.py --demo first)
Needs a model from pen_trainer.py (pen_model.pkl), next to this file.

Writing: end each item the same way you did in training - a short pause, or
hold the BOOT button / Space bar while writing it (set in the trainer with M).

Letters: if the model knows single letters, each word's letters are decoded
against a dictionary (pip install wordfreq for ~50k English words; put names
and your own terms in words.txt, one per line). A longer pause, or W, ends a word.

Keys
    Tab         wrong? cycle the last word through the next-best guesses
    1-5         pick guess n for the last word
    W           end the current word now (letters; otherwise a longer pause does it)
    [ ]         shorter / longer pause that starts a new word (letters)
    Backspace   remove the last item
    Enter       new line
    Space       pause mode: end the current item now / hold mode: hold while writing
    K           keep: add the last item (with its current label) to the training data
    S           save the text to transcript_<time>.txt
    C           clear        Esc  quit
"""
import argparse
import os
import queue
import threading
import time

import pygame

import pen_ml as ml

BG = (24, 27, 33)
FG = (225, 230, 240)
DIM = (150, 158, 175)
ACCENT = (140, 255, 170)
WARN = (255, 190, 110)
PAPER = (247, 245, 236)
INK = (40, 44, 60)
INK_UNSURE = (190, 120, 60)
RULE = (170, 200, 230)
MARGIN = (230, 150, 150)


class Live:
    def __init__(self, rx, source, model):
        pygame.init()
        pygame.display.set_caption("Pen - live text")
        self.screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE)
        mono = "consolas,menlo,dejavusansmono,monospace"
        self.f_small = pygame.font.SysFont(mono, 14)
        self.f = pygame.font.SysFont(mono, 17)
        self.f_mid = pygame.font.SysFont(mono, 22, bold=True)
        self.f_hand = pygame.font.SysFont("segoe print,bradley hand,comic sans ms,chalkboard,dejavusans", 34)
        self.rx, self.source, self.model = rx, source, model
        self.settings = ml.load_settings()
        self.fs = ml.FeatureStream()
        self.seg = ml.Segmenter(self.settings)
        self.letters = any(len(l) == 1 for l in model.labels)
        self.dictionary = ml.Dictionary() if self.letters else None
        self.tr = ml.Transcript(self.settings, self.dictionary)
        self.space = False
        self.last_rx = 0.0
        self.rx_times = []
        self.msg = ("", 0.0)
        self.flash = 0.0

    def say(self, text):
        self.msg = (text, time.time())

    def blit(self, text, pos, font=None, color=FG):
        s = (font or self.f).render(text, True, color)
        self.screen.blit(s, pos)
        return s.get_width()

    # ---------------------------------------------------------------- input
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYUP and e.key == pygame.K_SPACE:
                self.space = False
            if e.type != pygame.KEYDOWN:
                continue
            k = e.key
            if k == pygame.K_ESCAPE:
                return False
            if k == pygame.K_SPACE:
                self.space = True
                if self.settings["hold_mode"] < 0.5:
                    self.seg.force_end()
            elif k == pygame.K_TAB:
                self.tr.cycle()
            elif pygame.K_1 <= k <= pygame.K_5:
                self.tr.pick(k - pygame.K_1)
            elif k == pygame.K_BACKSPACE:
                self.tr.undo()
            elif k == pygame.K_RETURN:
                self.tr.add([("<enter>", 1.0)], None)
            elif k == pygame.K_w:
                self.tr.add([("<space>", 1.0)], None)       # end the current word now
            elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                d = 0.1 if k == pygame.K_RIGHTBRACKET else -0.1
                self.settings["word_gap_s"] = round(min(4.0, max(0.5, self.settings["word_gap_s"] + d)), 2)
                ml.save_settings(self.settings)
                self.say(f"A pause of {self.settings['word_gap_s']:.1f} s between letters now starts a new word")
            elif k == pygame.K_c:
                self.tr.clear()
            elif k == pygame.K_k and self.tr.items and self.tr.items[-1]["seg"] is not None:
                it = self.tr.items[-1]
                lab = ml.Transcript.label_of(it)
                ml.save_sample(it["seg"], lab, source="live")
                self.say(f"Saved this '{lab}' to the training data - retrain in pen_trainer.py to use it")
            elif k == pygame.K_s:
                name = os.path.join(ml.HERE, time.strftime("transcript_%Y%m%d_%H%M%S.txt"))
                try:
                    with open(name, "w", encoding="utf-8") as f:
                        f.write(self.tr.text(mark_unsure=False) + "\n")
                    self.say("Saved " + os.path.basename(name))
                except OSError as exc:
                    self.say(f"Couldn't save: {exc}")
        return True

    # ---------------------------------------------------------------- data
    def pump(self):
        for _ in range(600):
            try:
                s = self.rx.get_nowait()
            except queue.Empty:
                break
            self.last_rx = time.time()
            self.rx_times.append(self.last_rx)
            r = self.fs.push(s)
            if not r:
                continue
            seg = self.seg.push(r[0], r[1], hold=self.fs.button or self.space)
            if seg is not None:
                self.tr.add(self.model.predict(seg, top=999), seg)
                self.flash = time.time()
        self.rx_times = [x for x in self.rx_times if time.time() - x <= 1.0]

    # ---------------------------------------------------------------- drawing
    def wrap(self, text, font, width):
        lines = []
        for para in text.split("\n"):
            cur = ""
            for word in para.split(" "):
                trial = (cur + " " + word) if cur else word
                if font.size(trial)[0] <= width or not cur:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    def draw(self):
        w, h = self.screen.get_size()
        self.screen.fill(BG)
        tr = self.fs.tracker
        live = time.time() - self.last_rx < 1.0
        mode = "hold BOOT / Space while writing" if self.settings["hold_mode"] >= 0.5 else \
            f"pause {self.settings['end_quiet_s']:.1f} s after each item"
        rep = self.model.report
        top = (f"{self.source}  |  {len(self.rx_times)} Hz  |  model {rep.get('kind')} "
               f"{rep.get('cv_accuracy', 0):.0%}, {len(self.model.labels)} labels  |  items end: {mode}")
        self.blit(top if live else f"{self.source}  |  NO DATA - is the pen on and on this network?",
                  (14, 10), self.f_small, DIM if live else WARN)

        # the page
        px, py, pw, ph = 30, 40, w - 340, h - 110
        pygame.draw.rect(self.screen, (12, 13, 16), (px + 5, py + 5, pw, ph))
        pygame.draw.rect(self.screen, PAPER, (px, py, pw, ph))
        lh = self.f_hand.get_linesize() + 6
        y = py + 18 + lh
        while y < py + ph - 8:
            pygame.draw.line(self.screen, RULE, (px, y), (px + pw, y))
            y += lh
        pygame.draw.line(self.screen, MARGIN, (px + 60, py), (px + 60, py + ph))

        # render the transcript word by word so unsure words can be coloured
        x, y = px + 74, py + 18
        line_w = pw - 74 - 20
        tokens = self.tokens()
        for tok, unsure, last in tokens:
            if tok == "\n":
                x, y = px + 74, y + lh
                continue
            tw = self.f_hand.size(tok)[0]
            if x + tw > px + 74 + line_w and x > px + 74:
                x, y = px + 74, y + lh
            if y > py + ph - lh:
                break
            col = INK_UNSURE if unsure else INK
            if last and time.time() - self.flash < 0.6:
                pygame.draw.rect(self.screen, (255, 240, 170), (x - 3, y + 2, tw + 6, lh - 6), border_radius=4)
            self.screen.blit(self.f_hand.render(tok, True, col), (x, y))
            x += tw + self.f_hand.size(" ")[0]
        # cursor
        if int(time.time() * 2) % 2 and y < py + ph - lh:
            pygame.draw.line(self.screen, INK, (x, y + 8), (x, y + lh - 10), 2)

        # side panel
        sx = w - 290
        self.blit("Writing", (sx, 46), self.f, DIM)
        a = min(1.5, self.seg.activity())
        pygame.draw.rect(self.screen, (34, 38, 46), (sx, 70, 250, 14), border_radius=4)
        col = ACCENT if self.seg.state == "writing" else (110, 120, 140)
        pygame.draw.rect(self.screen, col, (sx, 70, int(250 * a / 1.5), 14), border_radius=4)
        self.blit("WRITING" if self.seg.state == "writing" else "waiting", (sx, 90), self.f, col)

        info = self.tr.last_info()
        if info:
            letters, opts, chosen = info
            y2 = 130
            if letters is not None:
                self.blit("Letters read", (sx, y2), self.f, DIM)
                self.blit(letters[:26], (sx, y2 + 22), self.f_mid, FG)
                y2 += 58
                self.blit("Word - best matches", (sx, y2), self.f, DIM)
            else:
                self.blit("Last item - guesses", (sx, y2), self.f, DIM)
            y2 += 26
            for i, (lab, p) in enumerate(opts[:5]):
                sel = i == chosen
                pygame.draw.rect(self.screen, (50, 55, 66), (sx + 130, y2 + 4, 90, 10), border_radius=3)
                pygame.draw.rect(self.screen, ACCENT if sel else (110, 120, 140),
                                 (sx + 130, y2 + 4, int(90 * p), 10), border_radius=3)
                self.blit(f"{i + 1} {lab[:12]}", (sx, y2), self.f, FG if sel else DIM)
                self.blit(f"{p:.0%}", (sx + 226, y2), self.f_small, DIM)
                y2 += 22
            self.blit("Tab / 1-5: pick another", (sx, y2 + 4), self.f_small, DIM)
        if self.letters:
            self.blit(f"New word: pause {self.settings['word_gap_s']:.1f} s or press W", (sx, h - 150),
                      self.f_small, DIM)
            self.blit("Dictionary: " + self.dictionary.source[:30], (sx, h - 132), self.f_small, DIM)

        keys = ["Tab/1-5 fix last word   Backspace remove last item   Enter new line   W end word",
                "K keep last as training data   S save text   C clear   [ ] word-gap pause   Esc quit"]
        for i, ln in enumerate(keys):
            self.blit(ln, (30, h - 56 + 20 * i), self.f_small, DIM)
        text, when = self.msg
        if text and time.time() - when < 4:
            self.blit(text, (30, h - 80), self.f, ACCENT)
        if not tr.cal_loaded and live:
            self.blit("Mount not calibrated - run pen_viewer.py and press K", (sx, h - 100), self.f_small, WARN)

    def tokens(self):
        """[(text, unsure, is_last)] in display order."""
        toks = self.tr.tokens()
        return [(w, u, i == len(toks) - 1) for i, (w, u) in enumerate(toks)]

    def run(self):
        clock = pygame.time.Clock()
        while self.handle_events():
            self.pump()
            self.draw()
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="Live handwriting recognition for the two-IMU pen")
    ml.add_source_args(ap)
    args = ap.parse_args()
    if args.demo:
        ml.use_demo_files()
    try:
        model = ml.Model.load()
    except Exception:
        raise SystemExit(f"No trained model ({os.path.basename(ml.MODEL_FILE)}). "
                         f"Run pen_trainer.py{' --demo' if args.demo else ''} and press T first.")
    rx = queue.Queue(maxsize=5000)
    stop = threading.Event()
    source, demo = ml.start_source(args, rx, stop, demo_labels=model.labels)
    try:
        Live(rx, source, model).run()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
