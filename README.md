<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff" alt="Sponsor Strata on GitHub Sponsors"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

> ### This is a fork: **Strata M1 Max**
>
> [catalinb13/Strata-M1-Max](https://github.com/catalinb13/Strata-M1-Max) is [Niko1221/Strata](https://github.com/Niko1221/Strata)
> tuned, measured and fixed on a **64 GB Apple M1 Max**, where the engine has to serve a 60-82 GB expert model from a
> machine that cannot hold it. The download buttons, the desktop app and the Terminal installers below are
> **upstream's**; this fork is built from source ([docs/MAC.md](docs/MAC.md)). It tracks upstream `main` — last merge:
> upstream 0.2.15 (`c77572806`), engine 0.1.60 — and carries its own engine and serving work on top; every measured
> change is written down (`docs/DETAILS.md` → *Apple silicon: M1 Max (this fork)*).

## Fork: what this is and what it does

A 64 GB Mac cannot hold a Qwen3.8-Flash-Next pack of 60-82 GB plus its K/V. This fork's Metal engine streams the
routed experts from the SSD into a resident arena and computes the dense half from memory, so the model runs at a
usable speed and answers identically to the non-streamed path.

Measured on **this** box (Apple M1 Max, 64 GB, macOS 27; Unsloth UD-IQ4_XS *huihui* pack, 204,800-token context,
`--kv fp16`, MTP drafter `--spec 2`, arena 30 GB):

| | measured |
| --- | ---: |
| decode, 12-18 K depth | 27-30 tok/s |
| decode, 68-82 K depth (live server, 261 requests) | **23.8 tok/s** median (p10 20.6, p90 26.8) |
| decode, 90-111 K depth | 21.7-24.7 tok/s |
| prefill | 320-374 tok/s at 4-5 K-token chunks |
| prompt tokens served from the prefix cache | 17.24 M of 18.15 M (**95 %**) |
| park → restore, 2.7-24 K-token conversation | 24 ms at 2.7 K tokens, 180 ms at 21.5 K (against minutes of re-prefill) |
| 142 K-token conversation reloaded from the session file | 1.6 s (4.1 GB) |
| draft acceptance | 88,613 of 134,700 (**65.8 %**) |
| first request after an engine start | 10-26 s (cold expert pages; a repeat costs 0.14 s) |

### What changed in the engine (`metal/`)

- **SSD-streamed expert arena.** 30 GB holds ~11,600 of the profile's top 24,576 expert pairs; the rest are read on
  demand with `F_NOCACHE` preads and handed to the kernel through slot tables. The prompt's own buffers borrow the
  arena while a prompt runs.
- **The decode window's cut handshake.** Each of the 48 per-layer cuts used an `MTLSharedEvent` pair, which costs
  230-400 µs because the queue stops and re-arms; a device-flag spin propagates in ~90 µs and never stops the queue.
  Window T=2: **87.8 → 56.0 ms**, decode **19.5-20.3 → 29.75-30.49 tok/s**, byte-identical output
  (`STRATA_METAL_CUT=event` opts back).
- **Read-ahead depth 12** for large-expert packs: 15.9 → 8.6 misses per window, window 88.8 → 83.0 ms.
- **`--kv fp16`** on Metal through direct-half kernels, and an MTP drafter that runs its draft round in one command
  buffer.
- **Correctness work the streaming path forced:** a non-finite-logits sentinel with checkpoint resume (#879), the
  batch window's M-RoPE table per row, `--kv fp16` in a batch window (wrote a scale plane out of bounds), and a
  session save that carries the drafter's K/V.
- **The prompt plan prices the arena.** It used to price the whole 56 GiB expert set as file tier and print "about
  385-421 GB read per 32 K-token prompt" — six times the measured reads. It now counts the arena's resident experts:
  *file set 30.4 GiB fits the page cache 43.0 GiB → about 0.0 GB read per 32 K-token prompt*.

### What changed in serving, from running long agent sessions on it

- **Conversation parking** keeps a switched-away conversation's K/V in RAM and restores it in tens of milliseconds.
  The Metal park used to re-copy the whole K/V on every turn (2.9 GB, 427-816 ms at 74-82 K tokens); it now reuses the
  retained prefix and copies only the tail — at 7 K tokens, 194 MB of a 573 MB snapshot reused, park 101.8 ms.
- **The auto-session file** is a 4 GB rewrite per save, so it now saves when the conversation grew by 4,096 tokens or
  the file is 10 minutes old, instead of 10 s after every request (that was 35.5 GiB of rewrites in five hours, with
  the server's queue held for up to 5.8 s each time).
- **Cache admission keeps the long conversation's slot**: a title/summary request of 2-7 K tokens no longer evicts a
  74 K-token conversation (28 evictions in five hours before).
- **Diagnostics that lied are fixed**: the expert-tier line printed all zeros 742 times because the streaming arena
  serves the experts, and a park's silent whole-conversation fallback now says which case fired.

### How it runs here

Source build (`docs/MAC.md`), then one command applies the tuning below to a config `metal/setup_mac.py` wrote, and
serves it (`--apply` writes, a backup lands beside it; `--start` serves):

```sh
metal/tools/m1max-serve.py --config strata-ud-iq4_xs-huihui.json --apply --start
```

Under it, `serve/server.py --engine strata --config <cfg>` with:

```text
--max-context 204800  --kv fp16  --prefill auto  --mtp <drafter> --spec 2 --suffix-draft 1
--expert-cache 1024   --conversation-cache-mib 8192 --conversation-cache-slots 6
--conversation-cache-min-tokens 2048                (auto-session on)
STRATA_METAL_STREAM=1  STRATA_METAL_ARENA_GB=30  STRATA_METAL_LEAVE_SHARE=0.35
STRATA_METAL_PROMPT_CHUNK=5120  STRATA_METAL_PROMPT_MOE=mpp  STRATA_METAL_GDN_SCAN=scan_v
STRATA_METAL_NFCHECK=1  STRATA_NAN_CKPT_RESUME=1  STRATA_METAL_F16=1
```

### What it costs, and the limits

- **Memory.** The engine plans the *process*, not the arena: 30.0 GB arena + 15.0-19.1 GB held before it (dense
  weights, K/V, graphs) + 0.5 GB n-gram table = 45.5-49.6 GB against a 44.7 GB cap, so the start line reads
  **OVER THE CAP** and says so. It has stayed up through every run here; a Mac short of memory can freeze.
- **Decode is bound by the expert handshake, not bandwidth** — the kernels run 85-152 GB/s against this box's
  measured 322.8 GB/s read ceiling. Feed knobs (prefetch depth, I/O threads, read split) all measured flat; the
  remaining weight is kernel-side, which is why several plausible ideas are written up as measured-and-rejected.
- **First request after a start** pays 10-26 s for cold expert pages. A warm-up does not fix it (it re-colds on the
  session restore), and that is documented rather than papered over.
- **Source fork, one machine.** No installer is published from here, and the numbers above come from one M1 Max; the
  app, the Terminal and the server are upstream's, and this fork is the engine and serve pieces listed above.

Where the evidence is: [`docs/DETAILS.md`](docs/DETAILS.md) (§ *Apple silicon: M1 Max (this fork)* — each arm with
its anchors), `gap/results/` for the probe runs, [`docs/MAC.md`](docs/MAC.md) for the build, and the commit log on
`metal-kv-fp16` for the change-by-change record.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.15-windows-x64-setup.exe` · `Strata-0.2.15-linux-x86_64.AppImage` · `Strata-0.2.15-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |
| **NVIDIA DGX Spark · Linux** | [DGX Spark setup](docs/DGX_SPARK_LINUX.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A GitHub sponsorship or a coffee helps fund development and testing on more hardware.

<a href="https://github.com/sponsors/Niko1221"><img src="https://img.shields.io/badge/Sponsor_on_GitHub-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors: sponsor Strata" height="48"></a>
<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

### Loyal supporters

Thank you to everyone who supports Strata.

<!-- supporters:start (one line per public supporter, newest last; copy a line and change the GitHub name in all four places) -->
<a href="https://github.com/ryan4yin"><img src="https://github.com/ryan4yin.png?size=96" width="48" height="48" alt="ryan4yin" title="ryan4yin"></a>
<a href="https://github.com/themactep"><img src="https://github.com/themactep.png?size=96" width="48" height="48" alt="themactep" title="themactep"></a>
<a href="https://github.com/macintog"><img src="https://github.com/macintog.png?size=96" width="48" height="48" alt="macintog" title="macintog"></a>
<a href="https://github.com/bisonbet"><img src="https://github.com/bisonbet.png?size=96" width="48" height="48" alt="bisonbet" title="bisonbet"></a>
<a href="https://github.com/vizv"><img src="https://github.com/vizv.png?size=96" width="48" height="48" alt="vizv" title="vizv"></a>
<a href="https://github.com/kod88vn"><img src="https://github.com/kod88vn.png?size=96" width="48" height="48" alt="kod88vn" title="kod88vn"></a>
<a href="https://github.com/seoulworker"><img src="https://github.com/seoulworker.png?size=96" width="48" height="48" alt="seoulworker" title="seoulworker"></a>
<a href="https://github.com/jackyini"><img src="https://github.com/jackyini.png?size=96" width="48" height="48" alt="jackyini" title="jackyini"></a>
<a href="https://github.com/router0mail"><img src="https://github.com/router0mail.png?size=96" width="48" height="48" alt="router0mail" title="router0mail"></a>
<!-- supporters:end -->

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

