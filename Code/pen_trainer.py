#!/usr/bin/env python3
"""
pen_trainer.py - teach the pen your words, letters and shapes.

Install:   pip install pygame numpy scikit-learn       (pyserial only for --serial)
Run:       python pen_trainer.py             (pen over WiFi, UDP 4210)
           python pen_trainer.py --ble       (pen over Bluetooth; pip install bleak)
           python pen_trainer.py --demo      (synthetic writing, no hardware)
Keep this file next to pen_ml.py and pen_viewer.py.

Workflow
  1. Calibrate the pen once in pen_viewer.py (K: mount, G: accelerometers).
  2. Here: Q  - quiet calibration (hold the pen still 2 s) so the trainer knows
                what "not writing" looks like.
  3. A  - add the words / letters / shapes you want (the vocabulary).
     Z  - adds all letters a-z at once (for letter-by-letter writing; the live app
          then decodes each word against a dictionary).
  4. R  - record. A word is shown; write it; it's captured automatically.
          Aim for 20-30 samples of each, written the way you'll really write
          them (on paper, normal size and speed, a little natural variety).
  5. T  - train. Shows the cross-validated accuracy and what gets confused.
  6. L  - live test: write, see what it recognises. Enter adds a correct
          result to the training data; 1-5 picks the right answer if it was wrong.
  Then use pen_live.py.

How an item ends (M switches):
  pause mode  write the item, then hold the pencil still ~0.6 s
  hold mode   hold the pen's BOOT button (needs the updated sketch) or the
              Space bar while you write each item; release when done
Your test page showed there are no natural pauses between words, so one of
these is needed - use the same mode for training and for live use.
"""
import argparse
import queue
import threading
import time

import numpy as np
import pygame

import pen_ml as ml

BG = (24, 27, 33)
FG = (225, 230, 240)
DIM = (150, 158, 175)
ACCENT = (140, 255, 170)
WARN = (255, 190, 110)
BAD = (255, 110, 100)
PANEL = (34, 38, 46)
TARGET = 25            # samples per label to aim for


class Trainer:
    def __init__(self, rx, source, demo):
        pygame.init()
        pygame.display.set_caption("Pen Trainer")
        self.screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE)
        mono = "consolas,menlo,dejavusansmono,monospace"
        self.f_small = pygame.font.SysFont(mono, 14)
        self.f = pygame.font.SysFont(mono, 17)
        self.f_mid = pygame.font.SysFont(mono, 24, bold=True)
        self.f_big = pygame.font.SysFont("segoe ui,helvetica,dejavusans,arial", 72, bold=True)
        self.rx, self.source, self.demo = rx, source, demo

        self.settings = ml.load_settings()
        self.fs = ml.FeatureStream()
        self.seg = ml.Segmenter(self.settings)
        self.vocab = ml.load_vocab()
        self.samples = ml.load_dataset()
        self.counts = ml.dataset_counts(self.samples)
        self.model = self._load_model()

        self.state = "home"
        self.sel = 0
        self.msg = ("", 0.0)
        self.space = False
        self.last_rx = 0.0
        self.rx_times = []
        self.confirm_x = 0.0

        self.prompt = None
        self.only = None
        self.last_saved = None
        self.last_seg = None
        self.session_n = 0
        self.quiet_g, self.quiet_a = [], []
        self.train_log = []
        self.train_result = None
        self.train_thread = None
        self.test_pred = None
        self.text_in = ""

    # ---------------------------------------------------------------- helpers
    def _load_model(self):
        try:
            return ml.Model.load()
        except Exception:
            return None

    def say(self, text):
        self.msg = (text, time.time())

    def refresh_counts(self):
        self.samples = ml.load_dataset()
        self.counts = ml.dataset_counts(self.samples)

    def next_prompt(self):
        pool = [self.only] if self.only else list(self.vocab)
        if not pool:
            self.prompt = None
            return
        least = min(self.counts.get(l, 0) for l in pool)
        cands = [l for l in pool if self.counts.get(l, 0) <= least + 2 and l != self.prompt] or pool
        self.prompt = str(np.random.choice(cands))
        if self.demo:
            self.demo.set_next(self.prompt)

    def blit(self, text, pos, font=None, color=FG, center=False):
        s = (font or self.f).render(text, True, color)
        r = s.get_rect()
        if center:
            r.center = pos
        else:
            r.topleft = pos
        self.screen.blit(s, r)
        return r

    # ---------------------------------------------------------------- input
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.TEXTINPUT and self.state == "addlabel":
                self.text_in += e.text
                continue
            if e.type == pygame.KEYUP and e.key == pygame.K_SPACE:
                self.space = False
            if e.type != pygame.KEYDOWN:
                continue
            k = e.key
            if k == pygame.K_SPACE:
                self.space = True
                if self.settings["hold_mode"] < 0.5:
                    self.seg.force_end()
                continue
            if self.state == "addlabel":
                self.key_addlabel(k)
            elif self.state == "home":
                if not self.key_home(k):
                    return False
            elif self.state == "record":
                self.key_record(k)
            elif self.state == "test":
                self.key_test(k)
            elif k in (pygame.K_ESCAPE, pygame.K_RETURN):
                if self.state != "train" or self.train_result is not None:
                    self.state = "home"
        return True

    def key_home(self, k):
        n = len(self.vocab)
        if k == pygame.K_ESCAPE:
            return False
        if k == pygame.K_UP and n:
            self.sel = (self.sel - 1) % n
        elif k == pygame.K_DOWN and n:
            self.sel = (self.sel + 1) % n
        elif k == pygame.K_z:
            added = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in self.vocab]
            self.vocab += added
            ml.save_vocab(self.vocab)
            self.say(f"Added {len(added)} letters a-z")
        elif k == pygame.K_a:
            self.state, self.text_in = "addlabel", ""
            pygame.key.start_text_input()
        elif k == pygame.K_DELETE and n:
            lab = self.vocab.pop(self.sel)
            ml.save_vocab(self.vocab)
            self.sel = min(self.sel, max(0, len(self.vocab) - 1))
            self.say(f"Removed '{lab}' from the vocabulary (its samples are kept)")
        elif k == pygame.K_x and n:
            lab = self.vocab[self.sel]
            if time.time() - self.confirm_x < 3:
                gone = sum(ml.delete_sample(s["path"]) for s in self.samples if s["label"] == lab)
                self.refresh_counts()
                self.confirm_x = 0
                self.say(f"Deleted {gone} samples of '{lab}'")
            else:
                self.confirm_x = time.time()
                self.say(f"Press X again to delete ALL samples of '{lab}'")
        elif k == pygame.K_q:
            self.state, self.quiet_g, self.quiet_a = "quiet", [], []
        elif k == pygame.K_m:
            self.settings["hold_mode"] = 0.0 if self.settings["hold_mode"] >= 0.5 else 1.0
            ml.save_settings(self.settings)
            self.seg.reset()
            self.say("Mode: " + ("HOLD - hold BOOT or Space while writing each item"
                                 if self.settings["hold_mode"] >= 0.5 else "PAUSE - pause after each item"))
        elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            d = 0.1 if k == pygame.K_RIGHTBRACKET else -0.1
            self.settings["end_quiet_s"] = round(min(2.0, max(0.2, self.settings["end_quiet_s"] + d)), 2)
            ml.save_settings(self.settings)
            self.say(f"Pause that ends an item: {self.settings['end_quiet_s']:.1f} s")
        elif k in (pygame.K_r, pygame.K_RETURN) and n:
            self.only = self.vocab[self.sel] if k == pygame.K_RETURN else None
            self.state, self.session_n, self.last_saved = "record", 0, None
            self.seg.reset()
            self.prompt = None
            self.next_prompt()
        elif k == pygame.K_t:
            self.start_training()
        elif k == pygame.K_l:
            if self.model is None:
                self.say("Train a model first (T)")
            else:
                self.state, self.test_pred = "test", None
                self.seg.reset()
        return True

    def key_addlabel(self, k):
        if k == pygame.K_RETURN:
            lab = self.text_in.strip()
            if lab and lab not in self.vocab:
                self.vocab.append(lab)
                ml.save_vocab(self.vocab)
                self.sel = len(self.vocab) - 1
                self.say(f"Added '{lab}'")
            pygame.key.stop_text_input()
            self.state = "home"
        elif k == pygame.K_ESCAPE:
            pygame.key.stop_text_input()
            self.state = "home"
        elif k == pygame.K_BACKSPACE:
            self.text_in = self.text_in[:-1]

    def key_record(self, k):
        if k == pygame.K_ESCAPE:
            self.state = "home"
            self.refresh_counts()
        elif k == pygame.K_BACKSPACE and self.last_saved:
            ml.delete_sample(self.last_saved[0])
            self.counts[self.last_saved[1]] = max(0, self.counts.get(self.last_saved[1], 1) - 1)
            self.say(f"Discarded the last '{self.last_saved[1]}'")
            self.prompt = self.last_saved[1]
            if self.demo:
                self.demo.set_next(self.prompt)
            self.last_saved = None
        elif k == pygame.K_s:
            self.next_prompt()

    def key_test(self, k):
        if k == pygame.K_ESCAPE:
            self.state = "home"
            self.refresh_counts()
            return
        if not self.test_pred:
            return
        seg, probs = self.test_pred
        label = None
        if k == pygame.K_RETURN:
            label = probs[0][0]
        elif pygame.K_1 <= k <= pygame.K_5 and k - pygame.K_1 < len(probs):
            label = probs[k - pygame.K_1][0]
        if label:
            ml.save_sample(seg, label, source="test")
            self.counts[label] = self.counts.get(label, 0) + 1
            self.say(f"Added to training data as '{label}' (press T later to retrain)")
            self.test_pred = None

    # ---------------------------------------------------------------- data
    def pump(self):
        for _ in range(600):
            try:
                s = self.rx.get_nowait()
            except queue.Empty:
                break
            now = time.time()
            self.last_rx = now
            self.rx_times.append(now)
            r = self.fs.push(s)
            if not r:
                continue
            t, x = r
            seg = self.seg.push(t, x, hold=self.fs.button or self.space)
            if self.state == "quiet":
                self.quiet_g.append(self.seg.gyro_lvl)
                self.quiet_a.append(self.seg.acc_lvl)
                if len(self.quiet_g) >= 200:
                    self.settings = ml.calibrate_quiet(self.quiet_g[40:], self.quiet_a[40:], self.settings)
                    self.seg.s = self.settings
                    ml.save_settings(self.settings)
                    self.say(f"Quiet levels set: writing starts above {self.settings['gyro_on']:.0f} deg/s "
                             f"or {self.settings['acc_on']:.2f} m/s^2")
                    self.state = "home"
            if seg is None:
                continue
            if self.state == "record" and self.prompt:
                path = ml.save_sample(seg, self.prompt)
                self.counts[self.prompt] = self.counts.get(self.prompt, 0) + 1
                self.last_saved = (path, self.prompt)
                self.last_seg = seg
                self.session_n += 1
                self.next_prompt()
            elif self.state == "test" and self.model is not None:
                self.test_pred = (seg, self.model.predict(seg))
                self.last_seg = seg
        self.rx_times = [x for x in self.rx_times if time.time() - x <= 1.0]

    def start_training(self):
        if self.train_thread and self.train_thread.is_alive():
            return
        self.refresh_counts()
        self.state, self.train_log, self.train_result = "train", [], None

        def work():
            try:
                m = ml.train(progress=self.train_log.append)
                m.save()
                self.model = m
                self.train_result = ("ok", m.report)
            except Exception as exc:
                self.train_result = ("error", str(exc))

        self.train_thread = threading.Thread(target=work, daemon=True)
        self.train_thread.start()

    # ---------------------------------------------------------------- drawing
    def status_line(self, w):
        tr = self.fs.tracker
        live = time.time() - self.last_rx < 1.0
        mode = "HOLD (BOOT / Space)" if self.settings["hold_mode"] >= 0.5 else \
            f"PAUSE ({self.settings['end_quiet_s']:.1f} s)"
        sens = "rear " + ("OK" if tr.valid[0] else "--") + "  tip " + ("OK" if tr.valid[1] else "--")
        txt = (f"{self.source}  |  {len(self.rx_times)} Hz  |  {sens}  |  "
               f"mount {'calibrated' if tr.cal_loaded else 'NOT calibrated (pen_viewer K)'}  |  item ends: {mode}")
        self.blit(txt if live else f"{self.source}  |  NO DATA - is the pen on and on this network?",
                  (14, 10), self.f_small, DIM if live else WARN)

    def meter(self, x, y, w, h):
        a = min(1.5, self.seg.activity())
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=4)
        col = ACCENT if self.seg.state == "writing" else (110, 120, 140)
        pygame.draw.rect(self.screen, col, (x, y, int(w * a / 1.5), h), border_radius=4)
        mx = x + int(w / 1.5)
        pygame.draw.line(self.screen, WARN, (mx, y - 3), (mx, y + h + 3), 2)
        btn = self.fs.button or self.space
        state = "WRITING" if self.seg.state == "writing" else "waiting"
        if self.settings["hold_mode"] >= 0.5:
            state += "   button " + ("HELD" if btn else "up")
        self.blit(state, (x + w + 12, y - 2), self.f, col)

    def mini_plot(self, seg, x, y, w, h):
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=4)
        if seg is None:
            return
        X, t = seg["X"], seg["t"]
        g = np.degrees(np.linalg.norm(X[:, 9:12], axis=1))
        a = np.linalg.norm(X[:, 0:3], axis=1)
        for sig, col in ((g / max(g.max(), 1e-6), (120, 170, 255)), (a / max(a.max(), 1e-6), (255, 170, 90))):
            pts = [(float(x + 4 + (w - 8) * ti / max(t[-1], 1e-6)), float(y + h - 4 - (h - 8) * v))
                   for ti, v in zip(t, sig)]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, col, False, pts, 2)
        self.blit(f"last capture {t[-1]:.2f} s   (blue rotation, orange acceleration)", (x + 6, y + 4),
                  self.f_small, DIM)

    def draw_home(self, w, h):
        self.blit("Pen Trainer", (14, 36), self.f_mid)
        y = 80
        self.blit("Vocabulary                     samples", (14, y), self.f, DIM)
        y += 26
        for i, lab in enumerate(self.vocab):
            n = self.counts.get(lab, 0)
            sel = i == self.sel
            if sel:
                pygame.draw.rect(self.screen, PANEL, (8, y - 3, 560, 25), border_radius=4)
            self.blit(("> " if sel else "  ") + lab[:26], (14, y), self.f, FG if sel else (200, 205, 215))
            bx = 330
            pygame.draw.rect(self.screen, (50, 55, 66), (bx, y + 4, 160, 12), border_radius=3)
            col = ACCENT if n >= TARGET else (WARN if n >= 10 else BAD)
            pygame.draw.rect(self.screen, col, (bx, y + 4, int(160 * min(1, n / TARGET)), 12), border_radius=3)
            self.blit(f"{n}", (bx + 170, y), self.f, col)
            if self.model and self.model.report.get("per_label", {}).get(lab) is not None:
                self.blit(f"{self.model.report['per_label'][lab]:.0%}", (bx + 215, y), self.f_small, DIM)
            y += 26
            if y > h - 230:
                self.blit("...", (14, y), self.f, DIM)
                break
        extra = sorted(set(self.counts) - set(self.vocab))
        if extra:
            self.blit("also in dataset: " + ", ".join(f"{l} ({self.counts[l]})" for l in extra)[:90],
                      (14, y + 4), self.f_small, DIM)

        x0 = 620
        self.blit("Model", (x0, 80), self.f, DIM)
        if self.model:
            r = self.model.report
            acc = r.get("cv_accuracy", 0)
            self.blit(f"{acc:.0%}", (x0, 104), self.f_big, ACCENT if acc >= 0.9 else (WARN if acc >= 0.7 else BAD))
            self.blit(f"cross-validated accuracy ({r.get('kind')}, {r.get('n_samples')} samples)",
                      (x0, 190), self.f_small, DIM)
            self.blit(f"trained {r.get('trained')}", (x0, 210), self.f_small, DIM)
            yy = 240
            for a, b, n in r.get("confused", [])[:5]:
                self.blit(f"'{a}' read as '{b}'  x{n}", (x0, yy), self.f_small, WARN)
                yy += 18
        else:
            self.blit("not trained yet", (x0, 104), self.f_mid, DIM)

        self.blit("Writing activity", (x0, h - 210), self.f, DIM)
        self.meter(x0, h - 182, 260, 16)

        keys = ["Up/Down select   A add label   Del remove label   X delete its samples",
                "Z add letters a-z   R record all   Enter record selected   T train   L live test",
                "Q quiet calibration   M pause/hold mode   [ ] pause length   Esc quit"]
        for i, ln in enumerate(keys):
            self.blit(ln, (14, h - 76 + 20 * i), self.f_small, DIM)

    def draw_record(self, w, h):
        self.blit(f"Recording{' - ' + self.only if self.only else ''}   ({self.session_n} this session)",
                  (14, 36), self.f_mid)
        if self.prompt:
            self.blit("Write:", (w // 2, h // 2 - 150), self.f_mid, DIM, center=True)
            self.blit(self.prompt, (w // 2, h // 2 - 80), self.f_big, FG, center=True)
            n = self.counts.get(self.prompt, 0)
            self.blit(f"{n} / {TARGET} samples", (w // 2, h // 2 - 20), self.f, DIM, center=True)
        how = ("Hold BOOT on the pen (or Space) while you write it, release when done."
               if self.settings["hold_mode"] >= 0.5 else
               "Write it, then hold the pencil still until it's captured.  (Space = done now)")
        self.blit(how, (w // 2, h // 2 + 20), self.f, ACCENT, center=True)
        self.meter(w // 2 - 200, h // 2 + 60, 300, 18)
        self.mini_plot(self.last_seg, w // 2 - 300, h // 2 + 100, 600, 110)
        if self.last_saved:
            self.blit(f"saved '{self.last_saved[1]}'  -  Backspace to discard it", (w // 2, h // 2 + 230),
                      self.f, DIM, center=True)
        self.blit("Esc back   S skip this word   Backspace discard last", (14, h - 30), self.f_small, DIM)

    def draw_quiet(self, w, h):
        self.blit("Quiet calibration", (14, 36), self.f_mid)
        self.blit("Hold the pencil still, as if you were about to write", (w // 2, h // 2 - 40),
                  self.f_mid, FG, center=True)
        p = min(1.0, len(self.quiet_g) / 200)
        pygame.draw.rect(self.screen, PANEL, (w // 2 - 200, h // 2 + 10, 400, 16), border_radius=4)
        pygame.draw.rect(self.screen, ACCENT, (w // 2 - 200, h // 2 + 10, int(400 * p), 16), border_radius=4)

    def draw_train(self, w, h):
        self.blit("Training", (14, 36), self.f_mid)
        y = 80
        for ln in self.train_log[-8:]:
            self.blit(ln, (14, y), self.f, DIM)
            y += 22
        if self.train_result is None:
            self.blit("working" + "." * (int(time.time() * 2) % 4), (14, y + 10), self.f, ACCENT)
            return
        kind, r = self.train_result
        if kind == "error":
            self.blit("Can't train yet: " + r, (14, y + 10), self.f, BAD)
        else:
            acc = r["cv_accuracy"]
            self.blit(f"{acc:.0%}", (14, y + 10), self.f_big, ACCENT if acc >= 0.9 else (WARN if acc >= 0.7 else BAD))
            self.blit(f"expected accuracy on new writing ({r['kind']}; "
                      f"extra-trees {r['cv_all']['forest']:.0%}, DTW {r['cv_all']['dtw']:.0%})",
                      (14, y + 96), self.f, DIM)
            yy = y + 130
            for lab, a in sorted(r["per_label"].items(), key=lambda kv: kv[1]):
                pygame.draw.rect(self.screen, (50, 55, 66), (180, yy + 4, 200, 12), border_radius=3)
                col = ACCENT if a >= 0.9 else (WARN if a >= 0.7 else BAD)
                pygame.draw.rect(self.screen, col, (180, yy + 4, int(200 * a), 12), border_radius=3)
                self.blit(lab[:18], (14, yy), self.f)
                self.blit(f"{a:.0%}  ({r['counts'].get(lab, 0)} samples)", (392, yy), self.f_small, DIM)
                yy += 22
                if yy > h - 60:
                    break
            xx = 620
            self.blit("Most confused", (xx, y + 130), self.f, DIM)
            for i, (a, b, n) in enumerate(r["confused"][:8]):
                self.blit(f"'{a}' read as '{b}' x{n}", (xx, y + 156 + 20 * i), self.f_small, WARN)
            if not r["confused"]:
                self.blit("nothing", (xx, y + 156), self.f_small, ACCENT)
            self.blit("Tips: record more samples of weak labels; very similar words need more.",
                      (14, h - 52), self.f_small, DIM)
        self.blit("Enter / Esc: back", (14, h - 30), self.f_small, DIM)

    def draw_test(self, w, h):
        self.blit("Live test", (14, 36), self.f_mid)
        self.meter(14, 80, 300, 16)
        if self.test_pred:
            seg, probs = self.test_pred
            top, p = probs[0]
            unsure = p < self.settings["reject_below"]
            self.blit(("? " if unsure else "") + top, (w // 2, h // 2 - 110), self.f_big,
                      WARN if unsure else FG, center=True)
            y = h // 2 - 40
            for i, (lab, pp) in enumerate(probs[:5]):
                pygame.draw.rect(self.screen, (50, 55, 66), (w // 2 - 60, y + 4, 240, 12), border_radius=3)
                pygame.draw.rect(self.screen, ACCENT if i == 0 else (110, 120, 140),
                                 (w // 2 - 60, y + 4, int(240 * pp), 12), border_radius=3)
                self.blit(f"{i + 1}  {lab[:16]}", (w // 2 - 260, y), self.f)
                self.blit(f"{pp:.0%}", (w // 2 + 190, y), self.f_small, DIM)
                y += 24
            self.mini_plot(seg, w // 2 - 300, y + 20, 600, 90)
            self.blit("Right?  Enter = yes, add it to the training data    1-5 = it was that one, add it",
                      (w // 2, y + 140), self.f, ACCENT, center=True)
        else:
            self.blit("Write something from your vocabulary", (w // 2, h // 2 - 40), self.f_mid, DIM, center=True)
        self.blit("Esc back   (retrain with T after adding samples)", (14, h - 30), self.f_small, DIM)

    def draw_addlabel(self, w, h):
        self.draw_home(w, h)
        pygame.draw.rect(self.screen, PANEL, (w // 2 - 300, h // 2 - 60, 600, 120), border_radius=8)
        self.blit("New label (word, letter, shape, or <space> <back> <enter>):", (w // 2 - 285, h // 2 - 48),
                  self.f_small, DIM)
        cursor = "_" if int(time.time() * 2) % 2 else " "
        self.blit(self.text_in + cursor, (w // 2 - 285, h // 2 - 15), self.f_mid)
        self.blit("Enter add   Esc cancel", (w // 2 - 285, h // 2 + 30), self.f_small, DIM)

    def draw(self):
        w, h = self.screen.get_size()
        self.screen.fill(BG)
        self.status_line(w)
        {"home": self.draw_home, "record": self.draw_record, "quiet": self.draw_quiet,
         "train": self.draw_train, "test": self.draw_test, "addlabel": self.draw_addlabel}[self.state](w, h)
        text, when = self.msg
        if text and time.time() - when < 4:
            s = self.f.render(text, True, ACCENT)
            self.screen.blit(s, (w - s.get_width() - 16, 36))

    def run(self):
        clock = pygame.time.Clock()
        while self.handle_events():
            self.pump()
            self.draw()
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="Train the pen to recognise your writing")
    ml.add_source_args(ap)
    args = ap.parse_args()
    rx = queue.Queue(maxsize=5000)
    stop = threading.Event()
    source, demo = ml.start_source(args, rx, stop)
    try:
        Trainer(rx, source, demo).run()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
