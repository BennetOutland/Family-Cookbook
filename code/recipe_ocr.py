#!/usr/bin/env python3
"""
Recipe OCR to Markdown Pipeline - Simple Version
Optimized for light-colored recipe cards with optional backgrounds
"""

import json
import re
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import cv2
from PIL import Image
import pytesseract
import requests


class RecipeOCRPipeline:
    def __init__(self, ollama_model="llama3.1:8b", ollama_url="http://localhost:11434"):
        """
        Initialize the recipe OCR pipeline
        
        Args:
            ollama_model: Name of the Ollama model to use
            ollama_url: URL of the Ollama API endpoint
        """
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
    
    def preprocess_image(self, image_path, debug=False):
        """
        Preprocess image for OCR - optimized for light cards with dark text
        
        Args:
            image_path: Path to the input image
            debug: Save intermediate preprocessing steps
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        print(f"Preprocessing image: {image_path}")
        
        # Read image with OpenCV
        img = cv2.imread(str(image_path))
        
        if debug:
            cv2.imwrite("debug_01_original.jpg", img)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if debug:
            cv2.imwrite("debug_02_grayscale.jpg", gray)
        
        # Light denoising to reduce background texture (like granite)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        if debug:
            cv2.imwrite("debug_03_denoised.jpg", denoised)
        
        # Use adaptive thresholding - works great for varied lighting and backgrounds
        # This creates a binary image (black text on white background)
        binary = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,  # Size of neighborhood for threshold calculation
            C=10           # Constant subtracted from mean
        )
        
        if debug:
            cv2.imwrite("debug_04_adaptive_threshold.jpg", binary)
        
        # Check if we need to invert (if background became black)
        # Count white vs black pixels
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)
        
        # If more black than white, we probably need to invert
        if black_pixels > white_pixels:
            binary = cv2.bitwise_not(binary)
            print("Inverted image (dark background detected)")
            if debug:
                cv2.imwrite("debug_05_inverted.jpg", binary)
        
        # Light morphological operations to clean up
        # Close small gaps in text
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        if debug:
            cv2.imwrite("debug_06_cleaned.jpg", cleaned)
        
        # Upscale if needed, but keep it reasonable
        height, width = cleaned.shape
        target_width = 2400  # Good balance for OCR
        
        if width < target_width:
            scale = target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Safety check
            if new_height > 5000:
                scale = 5000 / height
                new_width = int(width * scale)
                new_height = int(height * scale)
            
            cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"Upscaled image to {new_width}x{new_height}")
            
            if debug:
                cv2.imwrite("debug_07_upscaled.jpg", cleaned)
        
        return cleaned
    
    def extract_text_ocr(self, image_path, debug=False):
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image_path: Path to the input image
            debug: Save preprocessing debug images
            
        Returns:
            str: Extracted text
        """
        print("Running OCR...")
        
        # Preprocess image
        img = self.preprocess_image(image_path, debug=debug)
        
        # Configure Tesseract
        # PSM 6: Assume uniform block of text (good for recipe cards)
        # PSM 4: Assume single column of text
        # PSM 3: Fully automatic page segmentation
        
        # Try PSM 6 first (best for structured recipe cards)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        
        print(f"Extracted {len(text)} characters")
        
        # If result is suspiciously short, try PSM 3 (fully automatic)
        if len(text.strip()) < 50:
            print("Trying automatic page segmentation (PSM 3)...")
            custom_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(img, config=custom_config)
            print(f"Re-extracted {len(text)} characters")
        
        return text
    
    def clean_ocr_text(self, text):
        """
        Clean up common OCR errors
        
        Args:
            text: Raw OCR text
            
        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Fix common fraction issues
        text = text.replace('1/2', '½')
        text = text.replace('1/4', '¼')
        text = text.replace('3/4', '¾')
        text = text.replace('1/3', '⅓')
        text = text.replace('2/3', '⅔')
        
        # Common OCR errors
        text = text.replace('l/2', '½')
        text = text.replace('l/4', '¼')
        
        return text.strip()
    
    def extract_recipe_with_llm(self, ocr_text):
        """
        Use local LLM to extract structured recipe data
        
        Args:
            ocr_text: OCR-extracted text
            
        Returns:
            dict: Structured recipe data
        """
        print(f"Processing with LLM ({self.ollama_model})...")
        
        prompt = f"""You are extracting recipe information from OCR text. The OCR may have errors, so use your best judgment to correct obvious mistakes while staying faithful to the source.

OCR Text:
{ocr_text}

Extract the recipe information and return ONLY a valid JSON object with these exact fields:

- title (string): The recipe name/title - NOT the source or contest name. Look for the actual dish name (e.g., "Spicy Glazed Meatballs" not "Oatmeal Contest Winner")
- origin (string): Source/origin like "Second-prize winner in The 2005 Old Farmer's Almanac Reader Recipe Contest for oatmeal" (if mentioned, otherwise "Unknown")
- description (string): Brief description of what makes this dish special (empty string "" if not mentioned - do NOT invent one)
- servings (string): Number of servings (e.g., "4 to 6", or "Unknown" if not specified)
- prep_time (string): Preparation time (e.g., "15 min", or "Unknown" if not specified)
- cook_time (string): Cooking time (e.g., "30 min", or "Unknown" if not specified - look for baking/cooking times in instructions)
- total_time (string): Total time (calculate if you have prep + cook, otherwise "Unknown")
- ingredients (array of strings): List of ingredients with quantities - correct OCR errors (e.g., "7% cup" should be "⅓ cup" or "1/3 cup")
- ingredient_groups (object): If ingredients are clearly grouped with headers like "For the meatballs:", "For the glaze:", use this format: {{"Meatballs": [...], "Glaze": [...]}}. Otherwise leave empty.
- instructions (array of strings): Step-by-step cooking instructions as separate items
- notes (object): Only include keys that have actual content from the recipe:
  * make_ahead (string): How to prep in advance (only if mentioned)
  * substitutions (string): Alternative ingredients (only if mentioned)
  * storage (string): Storage instructions (only if mentioned)
  * tips (string): Helpful tricks (only if mentioned)
  * scaling (string): Doubling/halving notes (only if mentioned)
  * family_notes (string): Personal memories (only if mentioned)
- chefs_note (string): A personal tip or special instruction (empty string "" if none)

Important rules:
- The title should be the DISH NAME, not the source/contest name
- Only include information explicitly stated - no invention
- Correct obvious OCR errors in measurements (7% → ⅓, l/2 → ½, etc.)
- For ingredient_groups, look for section headers like "Meatballs" or "Glaze" or "For the..."
- Return empty string "" not placeholders like "[Brief description]"
- Return empty object {{}} for notes if there are no notes
- Return ONLY the JSON object, no other text
- Ensure valid JSON formatting"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            recipe_json = result.get("response", "{}")
            
            # Parse the JSON response
            recipe_data = json.loads(recipe_json)
            
            print("Successfully extracted recipe data")
            return recipe_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running (ollama serve)")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"Response was: {recipe_json}")
            raise
    
    def generate_markdown(self, recipe_data, output_path=None):
        """
        Generate markdown file from recipe data using cookbook template
        
        Args:
            recipe_data: Dictionary containing recipe information
            output_path: Optional path for output file
            
        Returns:
            str: Path to generated markdown file
        """
        print("Generating markdown...")
        
        # Extract data with defaults
        title = recipe_data.get('title', 'Untitled Recipe')
        origin = recipe_data.get('origin', recipe_data.get('source', 'Unknown'))
        servings = recipe_data.get('servings', 'Unknown')
        prep_time = recipe_data.get('prep_time', 'Unknown')
        cook_time = recipe_data.get('cook_time', 'Unknown')
        total_time = recipe_data.get('total_time', 'Unknown')
        description = recipe_data.get('description', '').strip()
        
        # Create markdown content with cookbook template
        markdown = f"""---
cssclass: cookbook
tags: recipe
---
# {title}

<div class="recipe-origin">
<strong>Origin:</strong> {origin}
</div>

<div class="recipe-meta">
<span><strong>Serves:</strong> {servings}</span>
<span><strong>Prep Time:</strong> {prep_time}</span>
<span><strong>Cook Time:</strong> {cook_time}</span>
<span><strong>Total Time:</strong> {total_time}</span>
</div>

"""
        
        # Only add description if it exists and is not a placeholder
        if description and description != '[Brief description of the dish]':
            markdown += f"{description}\n\n"
        
        markdown += "## Ingredients\n\n"
        
        # Add ingredients (handle grouped ingredients if present)
        ingredients = recipe_data.get('ingredients', [])
        ingredient_groups = recipe_data.get('ingredient_groups', {})
        
        if ingredient_groups:
            # Ingredients are organized into groups
            for group_name, group_items in ingredient_groups.items():
                markdown += f"\n**{group_name}**\n"
                for ingredient in group_items:
                    markdown += f"- {ingredient}\n"
        elif ingredients:
            # Simple ingredient list
            for ingredient in ingredients:
                markdown += f"- {ingredient}\n"
        
        markdown += "\n## Instructions\n\n"
        
        # Add instructions
        instructions = recipe_data.get('instructions', [])
        for i, instruction in enumerate(instructions, 1):
            # Try to format with bold action verb if not already formatted
            if instruction.strip() and not instruction.startswith('**'):
                # Try to extract first sentence/action as bold
                parts = instruction.split('.', 1)
                if len(parts) > 1:
                    markdown += f"{i}. **{parts[0].strip()}.** {parts[1].strip()}\n\n"
                else:
                    markdown += f"{i}. **{instruction.strip()}**\n\n"
            else:
                markdown += f"{i}. {instruction}\n\n"
        
        # Check if there are any actual notes
        notes = recipe_data.get('notes', {})
        has_notes = False
        
        if isinstance(notes, dict) and notes:
            # Check if any note values are non-empty
            has_notes = any(v and v.strip() for v in notes.values())
        elif isinstance(notes, str) and notes.strip():
            has_notes = True
        
        # Only add notes section if there are actual notes
        if has_notes:
            markdown += '<div class="notes-section">\n\n## Notes & Variations\n\n'
            
            if isinstance(notes, dict):
                if notes.get('make_ahead'):
                    markdown += f"- **Make-Ahead:** {notes['make_ahead']}\n"
                if notes.get('substitutions'):
                    markdown += f"- **Substitutions:** {notes['substitutions']}\n"
                if notes.get('storage'):
                    markdown += f"- **Storage:** {notes['storage']}\n"
                if notes.get('tips'):
                    markdown += f"- **Tips:** {notes['tips']}\n"
                if notes.get('scaling'):
                    markdown += f"- **Scaling:** {notes['scaling']}\n"
                if notes.get('family_notes'):
                    markdown += f"- **Family Notes:** {notes['family_notes']}\n"
            elif isinstance(notes, str):
                markdown += f"- **Tips:** {notes}\n"
            
            markdown += '\n</div>\n\n'
        
        # Add chef's note if present
        chefs_note = recipe_data.get('chefs_note', recipe_data.get('personal_note', '')).strip()
        if chefs_note:
            markdown += f'> **Chef\'s Note:** {chefs_note}\n'
        
        # Determine output path
        if output_path is None:
            # Create sanitized filename from title
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            output_path = f"{safe_title}.md"
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"Markdown saved to: {output_path}")
        return output_path
    
    def process_recipe(self, image_path, output_path=None, save_ocr=False, debug=False):
        """
        Complete pipeline: image → OCR → LLM → markdown
        
        Args:
            image_path: Path to recipe image
            output_path: Optional output path for markdown
            save_ocr: Whether to save raw OCR text
            debug: Whether to show debug output and save preprocessing steps
            
        Returns:
            str: Path to generated markdown file
        """
        print(f"\n{'='*60}")
        print(f"Processing recipe: {image_path}")
        print(f"{'='*60}\n")
        
        # Step 1: OCR
        ocr_text = self.extract_text_ocr(image_path, debug=debug)
        cleaned_text = self.clean_ocr_text(ocr_text)
        
        # Show OCR output in debug mode
        if debug:
            print("\n" + "="*60)
            print("DEBUG: OCR OUTPUT")
            print("="*60)
            print(cleaned_text)
            print("="*60 + "\n")
        
        # Always save OCR text for debugging
        ocr_path = Path(image_path).stem + "_ocr.txt"
        with open(ocr_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"OCR text saved to: {ocr_path}")
        
        # Step 2: LLM extraction
        recipe_data = self.extract_recipe_with_llm(cleaned_text)
        
        # Show extracted data in debug mode
        if debug:
            print("\n" + "="*60)
            print("DEBUG: EXTRACTED RECIPE DATA")
            print("="*60)
            print(json.dumps(recipe_data, indent=2))
            print("="*60 + "\n")
        
        # Step 3: Generate markdown with auto-filename from recipe title
        if output_path is None:
            # Use recipe title for filename
            title = recipe_data.get('title', 'recipe')
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            output_path = f"{safe_title}.md"
        
        markdown_path = self.generate_markdown(recipe_data, output_path)
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Recipe title: {recipe_data.get('title', 'Unknown')}")
        print(f"{'='*60}\n")
        
        return markdown_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert recipe images to structured markdown files using OCR and LLM"
    )
    parser.add_argument(
        "image",
        help="Path to recipe image file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output markdown file path (default: auto-generated from recipe title)"
    )
    parser.add_argument(
        "-m", "--model",
        default="llama3.1:8b",
        help="Ollama model to use (default: llama3.1:8b)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--save-ocr",
        action="store_true",
        help="Save raw OCR text to file (now always saved for debugging)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug output including OCR text, extracted JSON, and save preprocessing steps"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = RecipeOCRPipeline(
        ollama_model=args.model,
        ollama_url=args.ollama_url
    )
    
    # Process recipe
    try:
        markdown_path = pipeline.process_recipe(
            args.image,
            output_path=args.output,
            save_ocr=args.save_ocr,
            debug=args.debug
        )
        print(f"✓ Success! Recipe saved to: {markdown_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())