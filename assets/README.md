# Private-GPT Icon

This directory should contain the application icon for building the executable and installer.

## Required File

- **icon.ico**: Windows icon file (256x256 recommended, with multiple sizes embedded)

## Creating an Icon

### From PNG/SVG

Use an online converter or tools like:

```bash
# Using ImageMagick
convert logo.png -define icon:auto-resize=256,128,64,48,32,16 icon.ico

# Using GIMP
File → Export As → icon.ico
```

### Icon Requirements

- **Format**: ICO (Windows Icon)
- **Sizes**: 16x16, 32x32, 48x48, 256x256 (embedded in one file)
- **Transparency**: Supported (recommended)

### Temporary Solution

If you don't have an icon yet, the build will work without it (default icon will be used).

## Icon Locations

The icon is used in:
1. **Executable**: `build_exe.py` → `--windows-icon-from-ico=assets/icon.ico`
2. **Installer**: `installer/setup.iss` → `SetupIconFile=..\assets\icon.ico`
3. **Shortcuts**: Desktop and Start Menu shortcuts

## Design Tips

- Simple, recognizable design
- High contrast for visibility at small sizes
- Represents "privacy" and "AI" concepts
- Use brand colors consistently
