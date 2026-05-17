# Private-GPT Assets

This directory is reserved for application assets such as icons and screenshots.

## Current Build Path

The current repository builds with PyInstaller through:

```bash
uv run python build.py
```

At the moment, `build.py` and `PrivateGPT.spec` do not require an icon file.

## Optional Icon

If you add icon support later, place a Windows icon here:

```text
assets/icon.ico
```

Recommended embedded sizes:

- 16x16
- 32x32
- 48x48
- 256x256

Example conversion with ImageMagick:

```bash
convert logo.png -define icon:auto-resize=256,128,64,48,32,16 assets/icon.ico
```

After adding an icon, update the PyInstaller command in `build.py` and/or
`PrivateGPT.spec` to include it.
