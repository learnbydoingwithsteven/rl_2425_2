import fitz
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Open PDF
doc = fitz.open(r'c:\Users\wjbea\Downloads\learnbydoingwithsteven\rl_2425_2\reinforcement_learning_projects_2024-25.pdf')

print(f"Total pages: {len(doc)}")

# Extract all pages
for page_num in range(len(doc)):
    text = doc[page_num].get_text()
    with open(f'page_{page_num}.txt', 'w', encoding='utf-8') as f:
        f.write(f"=== PAGE {page_num} ===\n\n")
        f.write(text)
    print(f"Page {page_num}: {len(text)} characters")
    # Print first 200 chars to identify content
    print(text[:200].replace('\n', ' '))
    print()
