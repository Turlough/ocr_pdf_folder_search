# OCR PDF Folder Search

A Python tool for searching text content within PDF files across entire folder structures. This tool uses multi-threading to efficiently scan through multiple PDF files and extract text to find specific search terms.

## Features

- **Multi-threaded processing**: Scans multiple PDF files concurrently for faster performance
- **Recursive folder scanning**: Searches through all subdirectories automatically
- **Progress tracking**: Real-time progress bar showing scan completion status
- **Case-insensitive search**: Finds matches regardless of text case
- **Page-level results**: Shows which page contains the search term
- **UTF-8 support**: Handles international characters properly

## Requirements

- Python 3.6 or higher
- PyPDF2 >= 3.0.1
- tqdm >= 4.66.1

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install PyPDF2>=3.0.1 tqdm>=4.66.1
```

## Usage

Run the script:

```bash
python main.py
```

The program will prompt you for:
1. **Folder path**: Enter the directory you want to scan (supports drag-and-drop)
2. **Search term**: Enter the text you want to search for

### Example Output 
Enter the folder to scan
-> C:\Documents\PDFs
Enter the search term
-> invoice
Submitted 15 tasks
Scanning PDFs: 100%|██████████| 15/15 [00:05<00:00, 2.85it/s]
Found 3 results
Found "invoice". Page 2 in document1.pdf
Found "invoice". Page 1 in document2.pdf
Found "invoice". Page 5 in document3.pdf

## How It Works

1. **File Discovery**: The tool recursively walks through the specified folder and identifies all PDF files
2. **Multi-threading**: Uses `ThreadPoolExecutor` to process multiple PDF files simultaneously
3. **Text Extraction**: For each PDF, extracts text from all pages using PyPDF2
4. **Search**: Performs case-insensitive search for the specified term
5. **Results**: Returns file name, page number, and search term for each match

## Performance

- **Concurrent processing**: Multiple PDF files are processed simultaneously
- **Progress tracking**: Real-time feedback on scan progress
- **Memory efficient**: Processes files one at a time to minimize memory usage

## Limitations

- Only searches text content (not images or scanned documents)
- Requires PDFs to have extractable text content
- Search is case-insensitive but exact match only (no fuzzy search)

## Troubleshooting

### Common Issues

1. **No results found**: Ensure the PDF contains searchable text (not just scanned images)
2. **Permission errors**: Make sure you have read access to the folder and files
3. **Encoding issues**: The tool uses UTF-8 encoding for search terms

### Error Messages

- `FileNotFoundError`: Check that the folder path is correct
- `PermissionError`: Ensure you have read permissions for the folder
- `PyPDF2.errors.PdfReadError`: The PDF file may be corrupted or password-protected

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool.

## License

This project is open source and available under the MIT License. 