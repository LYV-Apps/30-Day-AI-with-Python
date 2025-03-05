# Setup Guide
- Download Python: python.org (or use Homebrew: `brew install python@3.13`)
- Verify: `python3 --version`
- Install pip: `python3 -m ensurepip --upgrade`

## Virtual Environments
When starting this project, we will create and use a virtual environment to manage dependencies:
1. Create: `python3 -m venv venv`
2. Activate:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
3. Install project libraries with `pip install -r requirements.txt`
4. Deactivate: `deactivate`