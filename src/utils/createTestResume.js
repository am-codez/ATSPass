import { PDFDocument, rgb, StandardFonts } from 'pdf-lib';

export const createTestResume = async () => {
  const pdfDoc = await PDFDocument.create();
  const page = pdfDoc.addPage();
  const { width, height } = page.getSize();

  // Load fonts
  const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
  const boldFont = await pdfDoc.embedFont(StandardFonts.HelveticaBold);

  // Define text content
  const content = [
    { text: 'John Doe', font: boldFont, size: 24 },
    { text: 'Software Engineer', font: font, size: 16 },
    { text: 'john.doe@email.com | (555) 123-4567', font: font, size: 12 },
    { text: 'San Francisco, CA', font: font, size: 12 },
    { text: '', font: font, size: 12 },
    { text: 'PROFESSIONAL SUMMARY', font: boldFont, size: 14 },
    { text: 'Experienced software developer with expertise in web development and system architecture. Skilled in creating scalable solutions and collaborating with cross-functional teams.', font: font, size: 12 },
    { text: '', font: font, size: 12 },
    { text: 'EXPERIENCE', font: boldFont, size: 14 },
    { text: 'Senior Software Engineer | Tech Solutions Inc. | 2019-Present', font: boldFont, size: 12 },
    { text: '• Developed and maintained web applications using React and Node.js', font: font, size: 12 },
    { text: '• Implemented scalable solutions for enterprise clients', font: font, size: 12 },
    { text: '• Led a team of 5 developers in delivering major projects', font: font, size: 12 },
    { text: '', font: font, size: 12 },
    { text: 'Software Developer | Innovation Labs | 2017-2019', font: boldFont, size: 12 },
    { text: '• Built responsive web applications using modern JavaScript frameworks', font: font, size: 12 },
    { text: '• Collaborated with design teams to implement user interfaces', font: font, size: 12 },
    { text: '', font: font, size: 12 },
    { text: 'EDUCATION', font: boldFont, size: 14 },
    { text: 'BS in Computer Science | University of Technology | 2017', font: font, size: 12 },
    { text: 'GPA: 3.8/4.0', font: font, size: 12 },
    { text: '', font: font, size: 12 },
    { text: 'SKILLS', font: boldFont, size: 14 },
    { text: '• Programming Languages: JavaScript, Python, Java', font: font, size: 12 },
    { text: '• Web Technologies: React, Node.js, Express', font: font, size: 12 },
    { text: '• Databases: MongoDB, PostgreSQL', font: font, size: 12 },
    { text: '• Tools: Git, Docker, AWS', font: font, size: 12 },
    { text: '• Soft Skills: Leadership, Problem Solving, Communication', font: font, size: 12 }
  ];

  // Draw text
  let y = height - 50;
  content.forEach(item => {
    if (item.text === '') {
      y -= 15;
    } else {
      page.drawText(item.text, {
        x: 50,
        y,
        size: item.size,
        font: item.font,
        maxWidth: width - 100
      });
      y -= item.size + 5;
    }
  });

  // Save the PDF
  const pdfBytes = await pdfDoc.save();
  return new Blob([pdfBytes], { type: 'application/pdf' });
}; 