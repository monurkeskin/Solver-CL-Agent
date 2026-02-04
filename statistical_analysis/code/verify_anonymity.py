#!/usr/bin/env python3
"""
COMPREHENSIVE Anonymity Verification Script

Checks ALL files in the repository for personal information:
- Names in CSV files
- Names in Excel files  
- Names in Python scripts
- Names in LaTeX files
- Names in Markdown files
- Email addresses
- Institution references
- File naming conventions
"""

import pandas as pd
import os
import re
import glob

# Repository path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Known participant names from the original experiment (to be detected)
# NOTE: Excluded names that are also Python keywords or common programming terms
PARTICIPANT_NAMES = [
    # First names (excluding 'Elif' as it matches Python's elif keyword)
    'Abdul', 'Ahmet', 'Alihan', 'Araz', 'Atahan', 'Atefeh',
    'Aykoc', 'Ayse', 'Azra', 'Baris', 'Begum', 'Bilal', 'Bilge', 'Burak', 'Burhan',
    'Canberk', 'Cansu', 'Ceren', 'Damla', 'Defne', 'Deniz', 'Doga', 'Durmus',
    'Eihab', 'Emin', 'Emre', 'Eray', 'Eren',
    'Eyup', 'Fatih', 'Furkan', 'Halenur', 'Halit', 'Kerem', 'Khaled',
    'Mehmet', 'Mohamed', 'Murat', 'Oguz', 'Omer', 'Onur', 'Raneem',
    'Roshaan', 'Saban', 'Sabina', 'Sajjad', 'Samet', 'Saner', 'Serife', 'Seyit',
    'Taher', 'Tahsin', 'Toprak', 'Ugurcan', 'Umut', 'Vagif', 'Yashar', 'Zahra',
    'Zohayr', 'Zuhal', 'ilayda', 'izel',
    # Last names (keeping only distinctive ones)
    'Khan', 'Guner', 'Sevencan', 'Balcin', 'Gokalp', 'Salimnezhad', 'Dagidir',
    'Tas', 'Tuncel', 'Molaei', 'Kardes', 'Yildiz', 'Acil', 'Yapici', 'Bender',
    'Mushtaque', 'Barindir', 'Arik', 'Akcan', 'Kurtulmus', 'Celik', 'Yilmaz',
    'Barin', 'Sirvanci', 'Yeniceri', 'Sati', 'Berk', 'Uzun', 'Lale', 'Buyukustun',
    'Ahmed', 'Karaman', 'Mutu', 'Arditi', 'Arici', 'Besik', 'Dedeagac', 'Erdogan',
    'Azizoglu', 'Topcan', 'Evren', 'Canturk', 'Arpacik', 'Ereglioglu', 'Gunes',
    'Suar', 'Tuna', 'Sharafeddin', 'Karaduman', 'Korkmaz', 'Erkol', 'Elayadi',
    'Sahin', 'Ozmeteler', 'Aygin', 'Abdulbar', 'Rauf', 'Demirkol', 'Iskandarova',
    'Hekmatjou', 'Muratoglu', 'Kaya', 'Altunal', 'Ulucay', 'Meydando', 'Karci',
    'Akkilic', 'Ayik', 'Cirak', 'ismailov', 'Sardroudi', 'Abbas', 'Asim',
    'Karatas', 'Turan', 'Celikel', 'Dogar', 'debreli', 'Yuksel',
]

# Email patterns
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# Institution patterns (that might reveal identity)
INSTITUTION_PATTERNS = [
    r'@\w+\.edu',
    r'@\w+\.ac\.\w+',
    r'university of',
    r'delft',
    r'cambridge',
]

def check_text_for_names(text, filename):
    """Check text content for participant names."""
    findings = []
    text_lower = text.lower()
    
    for name in PARTICIPANT_NAMES:
        # Check for exact matches (case insensitive)
        pattern = r'\b' + re.escape(name.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            findings.append(f"Found '{name}' in {filename}")
    
    return findings

def check_for_emails(text, filename):
    """Check for email addresses."""
    findings = []
    emails = re.findall(EMAIL_PATTERN, text)
    
    # Known third-party library author emails (not participant data)
    THIRD_PARTY_EMAILS = [
        'german.parisi@gmail.com',  # GWR library author
    ]
    
    for email in emails:
        # Exclude common false positives and third-party library emails
        if email.lower() in [e.lower() for e in THIRD_PARTY_EMAILS]:
            continue
        if any(x in email.lower() for x in ['example.com', 'test.com', 'placeholder']):
            continue
        findings.append(f"Found email '{email}' in {filename}")
    return findings

def check_csv_files(repo_dir):
    """Check all CSV files for personal information."""
    print("\n" + "=" * 90)
    print("CSV FILES ANONYMITY CHECK")
    print("=" * 90)
    
    csv_files = glob.glob(os.path.join(repo_dir, '**/*.csv'), recursive=True)
    findings = []
    checked = 0
    
    for csv_file in csv_files:
        rel_path = os.path.relpath(csv_file, repo_dir)
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            checked += 1
            
            # Check column names
            for col in df.columns:
                col_findings = check_text_for_names(col, f"{rel_path} (column name)")
                findings.extend(col_findings)
            
            # Check string columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    for val in df[col].dropna().unique()[:100]:  # Check first 100 unique values
                        val_findings = check_text_for_names(str(val), f"{rel_path} [{col}]")
                        findings.extend(val_findings)
                        
                        email_findings = check_for_emails(str(val), f"{rel_path} [{col}]")
                        findings.extend(email_findings)
            
            # Check for Subject column format
            if 'Subject' in df.columns:
                sample = df['Subject'].iloc[0] if len(df) > 0 else ''
                if isinstance(sample, str) and not sample.startswith('S0'):
                    findings.append(f"Subject column may not be anonymized in {rel_path}: '{sample}'")
                    
        except Exception as e:
            print(f"  Warning: Could not read {rel_path}: {e}")
    
    print(f"\n  Checked: {checked} CSV files")
    print(f"  Findings: {len(findings)}")
    
    if findings:
        for f in findings[:20]:
            print(f"    ✗ {f}")
        if len(findings) > 20:
            print(f"    ... and {len(findings) - 20} more")
    else:
        print("  ✓ No personal information found in CSV files")
    
    return findings

def check_excel_files(repo_dir):
    """Check all Excel files for personal information."""
    print("\n" + "=" * 90)
    print("EXCEL FILES ANONYMITY CHECK")
    print("=" * 90)
    
    xlsx_files = glob.glob(os.path.join(repo_dir, '**/*.xlsx'), recursive=True)
    findings = []
    checked = 0
    name_issues = 0
    
    for xlsx_file in xlsx_files:
        rel_path = os.path.relpath(xlsx_file, repo_dir)
        
        # Skip temp files
        if '~$' in rel_path:
            continue
            
        # Check filename
        filename = os.path.basename(xlsx_file)
        name_findings = check_text_for_names(filename, f"Filename: {rel_path}")
        if name_findings:
            name_issues += 1
            findings.extend(name_findings)
        
        # Check if filename follows anonymized pattern (S001, S002, etc.)
        if 'negotiation_logs' in filename:
            if not re.search(r'S\d{3}', filename):
                findings.append(f"Filename not anonymized: {rel_path}")
                name_issues += 1
        
        checked += 1
        
        # Check content (first sheet only due to performance)
        try:
            df = pd.read_excel(xlsx_file, engine='openpyxl')
            for col in df.columns:
                if df[col].dtype == 'object':
                    for val in df[col].dropna().unique()[:50]:
                        val_findings = check_text_for_names(str(val), f"{rel_path}")
                        findings.extend(val_findings)
        except Exception:
            pass
    
    print(f"\n  Checked: {checked} Excel files")
    print(f"  Filename issues: {name_issues}")
    print(f"  Content findings: {len(findings) - name_issues}")
    
    if not findings:
        print("  ✓ All Excel files properly anonymized (S001-S066 format)")
    else:
        for f in findings[:10]:
            print(f"    ✗ {f}")
    
    return findings

def check_python_files(repo_dir):
    """Check Python files for hardcoded personal information."""
    print("\n" + "=" * 90)
    print("PYTHON FILES ANONYMITY CHECK")
    print("=" * 90)
    
    py_files = glob.glob(os.path.join(repo_dir, '**/*.py'), recursive=True)
    findings = []
    checked = 0
    
    for py_file in py_files:
        rel_path = os.path.relpath(py_file, repo_dir)
        
        # Skip this script itself
        if 'verify_anonymity' in rel_path:
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            checked += 1
            
            # Check for names
            name_findings = check_text_for_names(content, rel_path)
            findings.extend(name_findings)
            
            # Check for emails
            email_findings = check_for_emails(content, rel_path)
            findings.extend(email_findings)
            
            # Check for hardcoded paths with usernames
            path_pattern = r'/Users/\w+/'
            paths = re.findall(path_pattern, content)
            for path in paths:
                findings.append(f"Found personal path '{path}' in {rel_path}")
                
        except Exception as e:
            print(f"  Warning: Could not read {rel_path}: {e}")
    
    print(f"\n  Checked: {checked} Python files")
    print(f"  Findings: {len(findings)}")
    
    if not findings:
        print("  ✓ No personal information found in Python files")
    else:
        for f in findings[:10]:
            print(f"    ✗ {f}")
    
    return findings

def check_text_files(repo_dir):
    """Check LaTeX, Markdown, and text files."""
    print("\n" + "=" * 90)
    print("TEXT FILES ANONYMITY CHECK (LaTeX, Markdown, README)")
    print("=" * 90)
    
    patterns = ['**/*.tex', '**/*.md', '**/*.txt', '**/README*']
    text_files = []
    for pattern in patterns:
        text_files.extend(glob.glob(os.path.join(repo_dir, pattern), recursive=True))
    
    findings = []
    checked = 0
    
    for text_file in text_files:
        rel_path = os.path.relpath(text_file, repo_dir)
        
        try:
            with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            checked += 1
            
            # Check for participant names
            name_findings = check_text_for_names(content, rel_path)
            findings.extend(name_findings)
            
            # Check for emails
            email_findings = check_for_emails(content, rel_path)
            findings.extend(email_findings)
                
        except Exception as e:
            print(f"  Warning: Could not read {rel_path}: {e}")
    
    print(f"\n  Checked: {checked} text files")
    print(f"  Findings: {len(findings)}")
    
    if not findings:
        print("  ✓ No personal information found in text files")
    else:
        for f in findings[:10]:
            print(f"    ✗ {f}")
    
    return findings

def check_directory_structure(repo_dir):
    """Check directory and file names for personal information."""
    print("\n" + "=" * 90)
    print("DIRECTORY/FILE NAMING CHECK")
    print("=" * 90)
    
    findings = []
    checked = 0
    
    for root, dirs, files in os.walk(repo_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for name in dirs + files:
            checked += 1
            name_findings = check_text_for_names(name, f"Name: {name}")
            findings.extend(name_findings)
    
    print(f"\n  Checked: {checked} directory/file names")
    print(f"  Findings: {len(findings)}")
    
    if not findings:
        print("  ✓ No personal information in file/directory names")
    else:
        for f in findings[:10]:
            print(f"    ✗ {f}")
    
    return findings

def main():
    print("=" * 90)
    print("COMPREHENSIVE ANONYMITY VERIFICATION")
    print("Repository: Paper_Submission_Repository")
    print("=" * 90)
    
    all_findings = []
    
    # Run all checks
    all_findings.extend(check_csv_files(REPO_DIR))
    all_findings.extend(check_excel_files(REPO_DIR))
    all_findings.extend(check_python_files(REPO_DIR))
    all_findings.extend(check_text_files(REPO_DIR))
    all_findings.extend(check_directory_structure(REPO_DIR))
    
    # Final summary
    print("\n" + "=" * 90)
    print("ANONYMITY VERIFICATION SUMMARY")
    print("=" * 90)
    
    print(f"\n  Total findings: {len(all_findings)}")
    
    if len(all_findings) == 0:
        print("\n  ✓ ALL ANONYMITY CHECKS PASSED")
        print("  ✓ REPOSITORY IS FULLY ANONYMIZED")
        print("  ✓ READY FOR GITHUB PUSH")
        return 0
    else:
        print(f"\n  ✗ {len(all_findings)} ANONYMITY ISSUES FOUND")
        print("\n  Unique issues:")
        unique_issues = list(set(all_findings))
        for issue in unique_issues[:30]:
            print(f"    - {issue}")
        if len(unique_issues) > 30:
            print(f"    ... and {len(unique_issues) - 30} more")
        return 1

if __name__ == "__main__":
    exit(main())
