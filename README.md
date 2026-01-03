# Family Cookbook

## How to Add Recipes

- Navigate to the assets folder 
- Upload an image to the recipe_images subfolder
- Follow the section for making markdown files


## How to Generate Markdown Files 

### Install and Setup Ollama 

The download is: https://ollama.com/download 

Then download a model:

```bash
# If this errors, ollama is probably already running
ollama serve 

# Pull the Llama model
ollama pull llama3.1:8b
```

### Create Virtual Environment

```bash
# Navigate to your project root
cd Family-Cookbook

# Create virtual environment
python3 -m venv venv
```

### Activate Virtual Environment

```bash
# Activate (you'll need to do this every time you open a new terminal)
source venv/bin/activate

# You should see (venv) at the start of your terminal prompt
```

### Install Dependencies

```bash
# Install all required packages (this may take a few minutes)
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test that everything installed correctly
python3 -c "import easyocr, cv2, requests; print('✓ All packages installed successfully')"
```

### Create Directories 

git does not allow pushing empty directories (this was done for space reasons), so you need to create them:

```bash
# Go into the assets folder
cd assets

# Make the image folders
mkdir recipe_images
mkdir processed_images
```

### Usage

Every time you want to use the scripts:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run your scripts
cd code/
python3 batch_process_recipes.py

# 3. When done, deactivate
deactivate
```

### Notes

- The virtual environment folder (`venv/`) is ignored by git
- First-time installation downloads EasyOCR models (~100MB) automatically
- Always activate the virtual environment before running scripts
- Make sure you have at least 15 GB free!


## Generating the Cookbook

- Download the Obsidian markdown editor and make a vault. 
- Clone this repo in the vault
- Go to Settings --> Appearance --> CSS Snippets and add the cookbook.css file from this repo and activate it.
- Install the Better Export PDF Package
- Then export with Better Export PDF!
