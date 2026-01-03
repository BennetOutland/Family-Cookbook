#!/usr/bin/env python3
"""
Batch Recipe Processor
Processes all recipe images in assets/recipe_images/ and organizes outputs
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import argparse


class BatchRecipeProcessor:
    def __init__(self, project_root=None, model="llama3.1:8b", debug=False):
        """
        Initialize the batch processor
        
        Args:
            project_root: Root directory of the project (auto-detected if None)
            model: Ollama model to use
            debug: Enable debug mode for OCR
        """
        # Auto-detect project root or use provided
        if project_root is None:
            # Assume script is in code/ folder
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        print(f"Project root: {self.project_root}")
        
        # Define folder structure
        self.code_dir = self.project_root / "code"
        self.assets_dir = self.project_root / "assets"
        self.recipe_images_dir = self.assets_dir / "recipe_images"
        self.markdown_dir = self.assets_dir / "markdown"
        self.processed_images_dir = self.assets_dir / "processed_images"
        
        # Path to the OCR script
        self.ocr_script = self.code_dir / "recipe_ocr.py"
        
        self.model = model
        self.debug = debug
        
        # Validate structure
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate that the expected folder structure exists"""
        print("\nValidating folder structure...")
        
        # Check code directory and script
        if not self.code_dir.exists():
            raise FileNotFoundError(f"Code directory not found: {self.code_dir}")
        
        if not self.ocr_script.exists():
            raise FileNotFoundError(f"OCR script not found: {self.ocr_script}\nMake sure recipe_ocr.py is in the code/ folder")
        
        print(f"✓ Found OCR script: {self.ocr_script}")
        
        # Check/create assets directories
        if not self.assets_dir.exists():
            print(f"Creating assets directory: {self.assets_dir}")
            self.assets_dir.mkdir(parents=True)
        
        if not self.recipe_images_dir.exists():
            print(f"Creating recipe_images directory: {self.recipe_images_dir}")
            self.recipe_images_dir.mkdir(parents=True)
        
        if not self.markdown_dir.exists():
            print(f"Creating markdown directory: {self.markdown_dir}")
            self.markdown_dir.mkdir(parents=True)
        
        if not self.processed_images_dir.exists():
            print(f"Creating processed_images directory: {self.processed_images_dir}")
            self.processed_images_dir.mkdir(parents=True)
        
        print("✓ Folder structure validated\n")
    
    def get_recipe_images(self):
        """
        Get list of image files in recipe_images directory
        
        Returns:
            list: List of Path objects for image files
        """
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        images = []
        for file_path in self.recipe_images_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                images.append(file_path)
        
        return sorted(images)
    
    def process_image(self, image_path):
        """
        Process a single recipe image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (success: bool, markdown_path: Path or None, error: str or None)
        """
        print(f"\n{'='*60}")
        print(f"Processing: {image_path.name}")
        print(f"{'='*60}")
        
        try:
            # Build command
            cmd = [
                "python3",
                str(self.ocr_script),
                str(image_path),
                "-m", self.model
            ]
            
            if self.debug:
                cmd.append("--debug")
            
            # Run the OCR script
            # Note: OCR script will create markdown in current directory
            result = subprocess.run(
                cmd,
                cwd=str(self.code_dir),  # Run from code directory
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = f"OCR script failed:\n{result.stderr}"
                print(f"✗ Error: {error_msg}")
                return False, None, error_msg
            
            # Find the generated markdown file
            # The OCR script creates it with the recipe title as filename
            # It will be in the code directory where we ran the script
            
            # Look for the most recently created .md file in code directory
            md_files = list(self.code_dir.glob("*.md"))
            if not md_files:
                # Also check for _ocr.txt files to see if OCR ran
                ocr_files = list(self.code_dir.glob("*_ocr.txt"))
                error_msg = f"No markdown file generated. OCR files found: {len(ocr_files)}"
                print(f"✗ Error: {error_msg}")
                return False, None, error_msg
            
            # Get the most recently modified .md file
            latest_md = max(md_files, key=lambda p: p.stat().st_mtime)
            
            # Move markdown to markdown directory
            final_md_path = self.markdown_dir / latest_md.name
            shutil.move(str(latest_md), str(final_md_path))
            print(f"✓ Markdown saved to: {final_md_path}")
            
            # Move image to processed_images directory
            final_image_path = self.processed_images_dir / image_path.name
            shutil.move(str(image_path), str(final_image_path))
            print(f"✓ Image moved to: {final_image_path}")
            
            # Clean up any debug files and OCR text files in code directory
            for pattern in ["debug_*.jpg", "*_ocr.txt"]:
                for debug_file in self.code_dir.glob(pattern):
                    debug_file.unlink()
            
            print(f"✓ Successfully processed {image_path.name}")
            return True, final_md_path, None
            
        except subprocess.TimeoutExpired:
            error_msg = "Processing timeout (>5 minutes)"
            print(f"✗ Error: {error_msg}")
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"✗ Error: {error_msg}")
            return False, None, error_msg
    
    def process_all(self):
        """
        Process all images in recipe_images directory
        
        Returns:
            dict: Summary of processing results
        """
        images = self.get_recipe_images()
        
        if not images:
            print(f"\nNo images found in {self.recipe_images_dir}")
            print("Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp")
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'results': []
            }
        
        print(f"\nFound {len(images)} image(s) to process\n")
        
        results = []
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}]", end=" ")
            
            success, md_path, error = self.process_image(image_path)
            
            results.append({
                'image': image_path.name,
                'success': success,
                'markdown': md_path.name if md_path else None,
                'error': error
            })
            
            if success:
                successful += 1
            else:
                failed += 1
        
        return {
            'total': len(images),
            'successful': successful,
            'failed': failed,
            'results': results
        }
    
    def print_summary(self, summary):
        """Print a summary of processing results"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images: {summary['total']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        
        if summary['failed'] > 0:
            print("\nFailed images:")
            for result in summary['results']:
                if not result['success']:
                    print(f"  ✗ {result['image']}: {result['error']}")
        
        if summary['successful'] > 0:
            print("\nSuccessful conversions:")
            for result in summary['results']:
                if result['success']:
                    print(f"  ✓ {result['image']} → {result['markdown']}")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process recipe images to markdown files"
    )
    parser.add_argument(
        "--project-root",
        help="Project root directory (auto-detected if not specified)",
        default=None
    )
    parser.add_argument(
        "-m", "--model",
        default="llama3.1:8b",
        help="Ollama model to use (default: llama3.1:8b)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (saves preprocessing images and shows OCR output)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create processor
        processor = BatchRecipeProcessor(
            project_root=args.project_root,
            model=args.model,
            debug=args.debug
        )
        
        # Process all images
        summary = processor.process_all()
        
        # Print summary
        processor.print_summary(summary)
        
        # Exit with error code if any failed
        if summary['failed'] > 0:
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())