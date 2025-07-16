#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Update this to the full path of your log file:
LOG_PATH = "training_log.txt"
# Prefix for the output images:
OUT_PREFIX = "plot"
# ────────────────────────────────────────────────────────────────────────────────

def parse_log(logfile):
    episodes, avg_returns, losses = [], [], []
    # Matches “episode #12345”
    ep_re = re.compile(r'episode #\s*(\d+)', re.IGNORECASE)
    # Matches a float like -0.0825
    float_re = re.compile(r'([-+]?\d*\.\d+|\d+)')

    with open(logfile, 'r') as f:
        for line in f:
            if 'avg_return' not in line or 'loss' not in line:
                continue
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue

            # part[1] → “…(episode #32040)”
            m_ep = ep_re.search(parts[1])
            # part[2] → “…avg_return ≃ -0.0825 ”
            m_ret = float_re.search(parts[2])
            # part[3] → “…loss=-0.2958”
            m_loss = re.search(r'loss\s*=\s*([-+]?\d*\.\d+|\d+)', parts[3])

            if not (m_ep and m_ret and m_loss):
                continue

            episodes.append(int(m_ep.group(1)))
            avg_returns.append(float(m_ret.group(1)))
            losses.append(float(m_loss.group(1)))

    return episodes, avg_returns, losses

def plot_series(x, y, xlabel, ylabel, title, out_file):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='.', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    eps, args, los = parse_log(LOG_PATH)

    # Apply the same downsampling to all three lists
    eps_downsampled = [eps[i] for i in range(len(eps)) if i % 1000 == 0]
    args_downsampled = [args[i] for i in range(len(args)) if i % 1000 == 0]
    los_downsampled = [los[i] for i in range(len(los)) if i % 1000 == 0]

    if not eps_downsampled: # Check if downsampled list is empty
        print(f"No matching lines found in {LOG_PATH} or not enough data for downsampling.")
        return

    plot_series(
        eps_downsampled, args_downsampled,
        "Episode", "Average Return", "Avg Return over Episodes",
        f"{OUT_PREFIX}_avg_return.png"
    )

    plot_series(
        eps_downsampled, los_downsampled, # Use the downsampled loss list here
        "Episode", "Loss", "Loss over Episodes",
        f"{OUT_PREFIX}_loss.png"
    )

if __name__ == "__main__":
    main()
