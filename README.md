🤖 **Personal AI Assistant: Your Web and PDF Guide**

![ezgif com-gif-maker (1)](https://github.com/Abhi0323/Generative-AI-based-Personal-Assistant/assets/112967999/8718ba7f-e075-4a42-bbef-9a6e94ff50a3)

Harness the power of AI to quickly get answers from multiple online sources and PDFs. Say goodbye to the era of skimming through countless web pages and dense documents!

**🚀 Features:**

**🌐 Multiple URL Querying:**

Input any number of URLs.
Pose your query.
Get amalgamated answers from all those sources.

**📄 PDF Uploading & Insightful Analysis:**

Swiftly upload PDFs.
Extract concise summaries without the redundancy.
Ask and get answers, all derived from the PDF's content.

**🔧 How Does It Work?:**
**Data Acquisition:**

**URLs**: Using LangChain's UnstructuredURLLoader, it efficiently processes and gathers data from web links.
**PDFs**: Leveraging PdfReader, the tool parses every page of the uploaded PDF, capturing all textual nuances.

**Text Segmentation:**

Content is strategically segmented into manageable chunks, ensuring optimality in both computation and response accuracy.
Text Embedding & Vector Storage:
Chunks undergo embedding, converting text to mathematical vectors while retaining semantic integrity. This is facilitated via OpenAIEmbeddings.
Vectors find their place in a FAISS vector database, enabling rapid similarity-based searching.

**Intelligent Query Handling:**

Upon a query, the tool sifts through the database, zeroing in on the most semantically relevant text chunks.
Human-like Response Generation:
The OpenAI Large Language Model (LLM) crafts precise, coherent answers from the selected chunks.

**PDF Summarization:**

A standout feature! The tool can synthesize an entire PDF into a crisp summary, presenting the document's crux.

**🌟 Benefits:**
Efficiency: Get direct answers and summaries in a jiffy.
Accuracy: Pooling data from diverse sources ensures comprehensive and accurate responses.
User Experience: Designed to be intuitive, catering to both tech-savvy and novice users.

**🌐 Applications:**
Research: Glean quick insights from academic papers to market research.
Reading: Short on time? Summarize long articles or reports.
Lifelong Learning: Stay curious, ask away, and get well-informed answers.
