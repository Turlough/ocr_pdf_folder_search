# import packages
import os
import concurrent.futures

import PyPDF2
from tqdm import tqdm


def extract_text_from_pdf(filepath, search_term):
    pdf = PyPDF2.PdfReader(filepath)
    # extract text and do the search
    for i, page in enumerate(pdf.pages):

        page_text = page.extract_text()
        if search_term.lower().decode('UTF-8') in page_text.lower():
            basename = os.path.basename(filepath)
            return (f'Found "{search_term.decode("utf-8")}". Page {i} in {basename}')
    
    return None

def scan_folder(folder, search_term, results_file):
    results = []
    # Walk through all subdirectories
    files_to_scan = []
    for root, _, files in os.walk(folder):
        files_to_scan.extend([os.path.join(root, f) for f in files if f.lower().endswith('.pdf')])
    
    # Use ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(extract_text_from_pdf, filepath, search_term): filepath 
            for filepath in files_to_scan
        }
        print(f'Submitted {len(future_to_file)} tasks')
        # Process results as they complete with tqdm
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                          total=len(files_to_scan), 
                          desc="Scanning PDFs"):
            result = future.result()
            if result:
                results.append(result)
                with open(results_file, 'a', encoding='utf-8') as f:
                    f.write(result + '\n')

    return results

if __name__ == "__main__":  
    folder = input('Enter the folder to scan\n\t-> ').replace('"', '').strip()
    search_term = input('Enter the search term\n\t-> ').lower().replace('"', '').strip().encode("utf-8")
    results_file = os.path.join(os.path.dirname(folder), f'{os.path.basename(folder)}_{search_term.decode("utf-8")}_search_results.txt')

    results = scan_folder(folder, search_term, results_file)
    
    if results:
        print(f'Found {len(results)} results')
        for result in results:
            print(result)
            # They have already been saved to the file as they are found
            print(f'Results saved to {results_file}')
    else:
        print('No results found')

