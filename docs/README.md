# EZfish Pipeline Documentation

This directory contains the Sphinx documentation for the EZfish Pipeline.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated documentation will be available in `_build/html/index.html`.

### View the Documentation

Open the built documentation in your browser:

```bash
# On Linux/Mac
open _build/html/index.html

# Or use Python's built-in server
cd _build/html
python -m http.server 8000
```

Then navigate to `http://localhost:8000` in your browser.

### Clean Build Files

To remove all built documentation:

```bash
make clean
```

## Documentation Structure

- `index.rst` - Main documentation landing page
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `pipeline_overview.rst` - Detailed pipeline documentation
- `api/` - API reference documentation
- `conf.py` - Sphinx configuration
- `_static/` - Static files (CSS, images, etc.)

## Theme

The documentation uses the [Furo](https://github.com/pradyunsg/furo) theme, which provides a clean and modern design similar to fairseq2.

## Contributing

When adding new modules or making significant changes:

1. Update the relevant `.rst` files
2. Add new API documentation in `api/`
3. Rebuild the documentation to check for errors
4. Ensure docstrings follow NumPy/Google style
