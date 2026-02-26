# min-add-nn-10

Minimal exact neural-network-style adder for two 10-digit strings.

## What it does

- Input: two 10-digit decimal strings
- Output: one 11-digit decimal string (includes overflow carry)
- Core model: 2-state recurrent carry machine implemented with `torch.nn`

## Files

- `minimal_adder_nn.py`: exact adder module
- `REPORT.md`: design notes and theoretical minimum discussion

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
python minimal_adder_nn.py
```
