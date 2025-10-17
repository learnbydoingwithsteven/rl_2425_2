import fitz
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Open PDF
doc = fitz.open(r'c:\Users\wjbea\Downloads\learnbydoingwithsteven\rl_2425_2\reinforcement_learning_projects_2024-25.pdf')

# Extract Project 2 pages (typically pages 3-6)
text = ""
for page_num in range(3, min(7, len(doc))):
    text += doc[page_num].get_text()

# Save to file
with open('project2_content.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("Project 2 content extracted successfully!")
print(f"Total characters: {len(text)}")
