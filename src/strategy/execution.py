from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TickerExecutionState:
    hold_policy: str = "keep"
    min_hold_days: int = 0
    reverse_confirm_days: int = 1

    current_position: int = 0
    days_held: int = 0
    pending_reverse_count: int = 0

    def update_signal(self, raw_signal: int) -> int:
        if raw_signal not in (-1, 0, 1):
            raw_signal = 0

        min_hold = max(0, int(self.min_hold_days))
        if self.current_position != 0 and self.days_held < min_hold:
            self.days_held += 1
            self.pending_reverse_count = 0
            return self.current_position

        effective_signal = raw_signal
        if raw_signal == 0:
            if str(self.hold_policy).lower() == "keep":
                effective_signal = self.current_position
            else:
                effective_signal = 0

        confirm_n = max(1, int(self.reverse_confirm_days))

        target = self.current_position
        if self.current_position == 0:
            if effective_signal in (-1, 1):
                target = effective_signal
            else:
                target = 0
            self.pending_reverse_count = 0
        else:
            if effective_signal == self.current_position:
                target = self.current_position
                self.pending_reverse_count = 0
            elif effective_signal == 0:
                target = 0
                self.pending_reverse_count = 0
            else:
                self.pending_reverse_count += 1
                if self.pending_reverse_count >= confirm_n:
                    target = effective_signal
                    self.pending_reverse_count = 0
                else:
                    target = self.current_position

        if target != self.current_position:
            self.current_position = int(target)
            self.days_held = 1 if self.current_position != 0 else 0
        else:
            if self.current_position != 0:
                self.days_held += 1
            else:
                self.days_held = 0

        return int(self.current_position)
