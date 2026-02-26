# Minimal Neural Network for Adding Two 10-Digit Strings

## 1) Problem statement (made precise)
Inputs:
- `A`: 10-character digit string (`0-9`)
- `B`: 10-character digit string (`0-9`)

Output (choose one explicitly):
- **Full sum**: up to 11 digits (`0` to `19999999998`)
- **Fixed 10 digits**: sum modulo `10^10`

Important: adding two 10-digit numbers can overflow to 11 digits. If you require exactly 10 output digits, you are defining modulo arithmetic.

## 2) Key correction to your neuron intuition
Your idea was:
- 10 neurons at input
- 10 neurons at output
- middle layer grows to 20 for carry

For string-digit addition, this is not the right scaling.

- Carry is only **1 bit** (`0` or `1`) at each position.
- So the required internal memory does **not** grow with number length (`N`).
- The algorithm is `O(N)` in time, but `O(1)` in memory.

So for `N=10`, the middle state is not 20 carries; it is one carry state reused across all 10 steps.

## 3) Theoretical minimum (state complexity)
If you process from least significant digit to most significant digit, each step needs only:
- current digits `a_t`, `b_t`
- previous carry `c_t in {0,1}`

Update:
- `s_t = a_t + b_t + c_t`
- output digit: `y_t = s_t mod 10`
- next carry: `c_{t+1} = 1[s_t >= 10]`

This is a 2-state transducer (`carry=0` or `carry=1`).

### Lower bound
An exact adder must distinguish at least two internal contexts (incoming carry 0 vs 1), so at least 1 bit of state is necessary.

### Upper bound
A recurrent model with 1-bit carry state is sufficient.

So, information-theoretically, the minimal hidden memory is:
- **1 binary state unit** (or equivalently 2 one-hot state neurons).

## 4) What “input/output neurons” really means here
There are two common interpretations:

1. **Token-per-step (recommended)**
   - Per time step input is one digit from each number.
   - If one-hot encoded: 10 + 10 = **20 input channels** per step.
   - Output is one digit: **10-way output** per step.
   - Reuse same cell for 10 steps.

2. **All digits at once (not minimal)**
   - Two full one-hot strings at once: `2 * 10 * 10 = 200` input units.
   - Output one-hot string: `10 * 10 = 100` output units.
   - Usually needs a deeper network to emulate carry chain.

Your “10 at head, 10 at end” is only true for one single digit token, not whole 10-digit strings simultaneously.

## 5) Smallest achieved: what can be claimed rigorously
There is no widely accepted benchmark with a single canonical “world-record smallest neuron count” for *exactly* this task setup.

What can be claimed rigorously:
- **Achievable by construction**: exact 10-digit addition with a 2-state recurrent carry machine (1 bit of hidden state).
- In neural terms, with hard-threshold style units, this can be implemented exactly.
- In trainable smooth networks (RNN/GRU/LSTM), practical hidden size is usually larger than theoretical minimum because optimization is imperfect.

So the strongest precise statement is:
- **Theoretical smallest hidden memory: 1 bit**.
- **Constructively achievable exact algorithm: yes** (state machine / recurrent cell).
- **Smallest trainable-from-scratch hidden width: setup-dependent; no universal published minimum for this exact spec.**

## 6) Minimal architecture blueprint (for your project)
Use right-to-left processing (reverse strings).

Per step:
1. Read `(a_t, b_t)`
2. Compute `s_t = a_t + b_t + carry`
3. Emit digit `s_t % 10`
4. Update `carry = 1 if s_t >= 10 else 0`

After 10 steps:
- If full-sum mode and `carry=1`, prepend final `1`.
- If fixed-width mode, drop overflow carry.

This is the exact adder core you can later generalize to any `N` without changing hidden-state size.

## 7) Generalization to N digits
For N-digit addition with same digit vocabulary:
- Hidden memory requirement stays constant (carry bit).
- Computation steps grow linearly with `N`.
- So scaling is by sequence length, not hidden width.

