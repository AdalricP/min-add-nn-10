"""Exact 10-digit + 10-digit adder with 2-state recurrent carry."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CarryAdderCell(nn.Module):
    """Single addition step over one digit pair plus carry state."""

    def __init__(self) -> None:
        super().__init__()

        self.next_carry = nn.Embedding(200, 2)
        self.digit_head = nn.Embedding(200, 10)

        next_carry_table = torch.zeros(200, 2, dtype=torch.float32)
        digit_table = torch.zeros(200, 10, dtype=torch.float32)

        for carry in (0, 1):
            for a in range(10):
                for b in range(10):
                    idx = carry * 100 + a * 10 + b
                    total = a + b + carry
                    out_digit = total % 10
                    out_carry = total // 10

                    next_carry_table[idx, out_carry] = 1.0
                    digit_table[idx, out_digit] = 1.0

        with torch.no_grad():
            self.next_carry.weight.copy_(next_carry_table)
            self.digit_head.weight.copy_(digit_table)

        self.next_carry.weight.requires_grad_(False)
        self.digit_head.weight.requires_grad_(False)

    def forward(
        self,
        a_t: torch.Tensor,
        b_t: torch.Tensor,
        h_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return output digit logits and next carry state."""
        carry_idx = h_t.argmax(dim=-1)
        pair_idx = a_t * 10 + b_t
        table_idx = carry_idx * 100 + pair_idx

        digit_logits = self.digit_head(table_idx)
        h_next = self.next_carry(table_idx)
        return digit_logits, h_next


class MinimalAdderNN(nn.Module):
    """Exact fixed-width string adder returning 11 output digits."""

    def __init__(self, num_digits: int = 10) -> None:
        super().__init__()
        self.num_digits = num_digits
        self.cell = CarryAdderCell()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Add digit tensors of shape [batch, num_digits]."""
        if a.shape != b.shape:
            raise ValueError(f"Input shapes must match, got {a.shape} vs {b.shape}")
        if a.ndim != 2 or a.shape[1] != self.num_digits:
            raise ValueError(
                f"Expected [batch, {self.num_digits}] digits, got {tuple(a.shape)}"
            )

        batch = a.shape[0]
        device = a.device

        h_t = F.one_hot(torch.zeros(batch, dtype=torch.long, device=device), num_classes=2)
        h_t = h_t.to(dtype=torch.float32)

        lsd_first_logits: list[torch.Tensor] = []

        for pos in range(self.num_digits - 1, -1, -1):
            step_logits, h_t = self.cell(a[:, pos], b[:, pos], h_t)
            lsd_first_logits.append(step_logits)

        result_10 = torch.stack(list(reversed(lsd_first_logits)), dim=1)

        leading_digit = h_t.argmax(dim=-1)
        leading_logits = F.one_hot(leading_digit, num_classes=10).to(dtype=torch.float32)
        leading_logits = leading_logits.unsqueeze(1)

        return torch.cat([leading_logits, result_10], dim=1)

    @staticmethod
    def encode_strings(nums: Iterable[str]) -> torch.Tensor:
        """Encode fixed-length digit strings into tensor [batch, num_digits]."""
        rows: List[List[int]] = []
        for s in nums:
            if not s.isdigit():
                raise ValueError(f"Non-digit input: {s!r}")
            rows.append([int(ch) for ch in s])
        return torch.tensor(rows, dtype=torch.long)

    @staticmethod
    def decode_logits(logits: torch.Tensor) -> List[str]:
        """Decode logits [batch, steps, 10] into digit strings."""
        digits = logits.argmax(dim=-1)
        out: List[str] = []
        for row in digits.tolist():
            out.append("".join(str(d) for d in row))
        return out


if __name__ == "__main__":
    model = MinimalAdderNN(num_digits=10)

    a_str = ["0000000001", "9999999999", "1234567890", "5000000000"]
    b_str = ["0000000009", "0000000001", "8765432109", "5000000000"]

    a = model.encode_strings(a_str)
    b = model.encode_strings(b_str)

    logits = model(a, b)
    out = model.decode_logits(logits)

    for x, y, z in zip(a_str, b_str, out):
        print(f"{x} + {y} = {z}")
