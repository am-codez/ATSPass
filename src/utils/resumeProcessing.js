import { PDFDocument } from 'pdf-lib';
import * as pdfjsLib from 'pdfjs-dist';

// Initialize PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

export const processResume = async (file) => {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
    let fullText = '';
    
    // Process each page
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      
      // Sort text items by vertical position (y) and then horizontal position (x)
      const sortedItems = textContent.items.sort((a, b) => {
        // If y positions are significantly different (more than 5 units), sort by y
        if (Math.abs(a.transform[5] - b.transform[5]) > 5) {
          return b.transform[5] - a.transform[5];
        }
        // Otherwise sort by x position
        return a.transform[4] - b.transform[4];
      });

      let currentLine = '';
      let currentY = null;
      let pageText = '';

      // Process text items to preserve formatting
      for (const item of sortedItems) {
        const y = item.transform[5];
        
        // If this is a new line (y position changed significantly)
        if (currentY === null || Math.abs(y - currentY) > 5) {
          if (currentLine) {
            pageText += currentLine.trim() + '\n';
          }
          currentLine = item.str;
        } else {
          // Same line, add space if needed
          currentLine += (currentLine ? ' ' : '') + item.str;
        }
        currentY = y;
      }

      // Add the last line
      if (currentLine) {
        pageText += currentLine.trim();
      }

      // Clean up the page text
      pageText = pageText
        .replace(/\s+/g, ' ') // Replace multiple spaces with single space
        .replace(/\n\s+/g, '\n') // Remove leading spaces from lines
        .trim();

      // Add page break if not the last page
      fullText += pageText + (i < pdf.numPages ? '\n\n' : '');
    }
    
    // Clean up the text while preserving important formatting
    return fullText
      .replace(/\n{3,}/g, '\n\n') // Replace multiple newlines with double newlines
      .replace(/\s+/g, ' ') // Replace multiple spaces with single space
      .trim();
  } catch (error) {
    console.error('Error processing PDF:', error);
    throw new Error('Failed to process PDF file. Please make sure it is a valid PDF.');
  }
};

export const createOptimizedPDF = async (text) => {
  try {
    const pdfDoc = await PDFDocument.create();
    const page = pdfDoc.addPage();
    const { width, height } = page.getSize();
    
    // Add text to the PDF
    page.drawText(text, {
      x: 50,
      y: height - 50,
      size: 12,
      maxWidth: width - 100,
    });

    // Generate PDF bytes
    const pdfBytes = await pdfDoc.save();
    return new Blob([pdfBytes], { type: 'application/pdf' });
  } catch (error) {
    console.error('Error creating PDF:', error);
    throw new Error('Failed to create optimized PDF: ' + error.message);
  }
}; 