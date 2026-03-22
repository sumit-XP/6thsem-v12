"""
YOLOv12 Project Report - Combine All Chapters Into Single Document

This script combines all chapter files into a single complete report document.
"""

import os
from pathlib import Path

# Define paths
REPORT_DIR = Path(__file__).parent / "report"
OUTPUT_FILE = REPORT_DIR / "Complete_YOLOv12_Report.md"

# Chapter files in order
CHAPTERS = [
    "Chapter_01_Abstract_Introduction.md",
    "Chapter_02_Literature_Review.md",
    "Chapter_03_System_Design.md",
    "Chapter_04_Implementation.md",
    "Chapter_05_Results.md",
    "Chapter_06_Conclusion.md",
    "References.md",
    "Appendices.md"
]

def combine_chapters():
    """Combine all chapter files into a single document."""
    
    print("=" * 70)
    print("YOLOv12 Project Report - Chapter Combiner")
    print("=" * 70)
    print()
    
    if not REPORT_DIR.exists():
        print(f"❌ Error: Report directory not found: {REPORT_DIR}")
        return
    
    combined_content = []
    
    # Add title page
    combined_content.append("# Automated Dairy Cow Behavior Recognition")
    combined_content.append("# Using YOLOv12 Augmented Deep Learning Architecture")
    combined_content.append("")
    combined_content.append("## Complete Project Report")
    combined_content.append("")
    combined_content.append("---")
    combined_content.append("")
    combined_content.append("**5th Semester Mini Project**")
    combined_content.append("")
    combined_content.append("**Architecture**: YOLOv12 Augmented")
    combined_content.append("")
    combined_content.append("**Performance**: 93.1% mAP@0.5, 8.5ms inference")
    combined_content.append("")
    combined_content.append("---")
    combined_content.append("")
    combined_content.append("\\newpage")
    combined_content.append("")
    
    # Combine chapters
    for i, chapter_file in enumerate(CHAPTERS, 1):
        chapter_path = REPORT_DIR / chapter_file
        
        if not chapter_path.exists():
            print(f"⚠️  Warning: {chapter_file} not found, skipping...")
            continue
        
        print(f"✓ Processing {chapter_file}...")
        
        with open(chapter_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        combined_content.append(content)
        combined_content.append("")
        combined_content.append("\\newpage")  # Page break for PDF export
        combined_content.append("")
    
    # Write combined file
    print()
    print(f"Writing combined report to: {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_content))
    
    # Statistics
    total_lines = len(combined_content)
    total_words = sum(len(line.split()) for line in combined_content)
    file_size_kb = OUTPUT_FILE.stat().st_size / 1024
    
    print()
    print("=" * 70)
    print("✅ Report Successfully Combined!")
    print("=" * 70)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total lines: {total_lines:,}")
    print(f"Total words: {total_words:,}")
    print(f"File size: {file_size_kb:.1f} KB")
    print()
    print("Next steps:")
    print("  1. Open Complete_YOLOv12_Report.md in a Markdown viewer")
    print("  2. Export to PDF using:")
    print("     - Pandoc: pandoc Complete_YOLOv12_Report.md -o Report.pdf --toc")
    print("     - Typora: File → Export → PDF")
    print("     - VS Code: Right-click → Markdown PDF: Export")
    print()

if __name__ == "__main__":
    try:
        combine_chapters()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
