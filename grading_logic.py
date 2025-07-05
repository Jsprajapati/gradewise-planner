import os
import re
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import tempfile
import shutil

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# LangGraph for workflow
from langgraph.graph import StateGraph, START, END


from os import getenv
from dotenv import load_dotenv

load_dotenv()

# Rich for better console output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Optional dependencies - will be used if available
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    HUGGINGFACE_AVAILABLE = True
    print("Huggingface available")
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Constants
_output_path = Path(__file__).parent / "output_res" 
GOOGLE_API_KEY = getenv("GOOGLE_API_KEY")
DEFAULT_MODEL = "gemini-2.0-flash-001"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500
TEMPERATURE = 0.2

class AIAssignmentEvaluator:
    def __init__(self, api_key=None, model=DEFAULT_MODEL, verbose=False):
        """Initialize the Assignment Evaluator with the specified LLM."""
        self.api_key = api_key or GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key must be provided or set as GOOGLE_API_KEY environment variable")
        
        self.model = model
        self.verbose = verbose
        self.console = Console() if RICH_AVAILABLE else None
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=TEMPERATURE,
            Google_api_key=self.api_key,
            max_tokens=4000
        )
        
        # Initialize AI content detector if available
        self.ai_detector = None
        if HUGGINGFACE_AVAILABLE:
            try:
                self.ai_detector = self._setup_ai_detector()
            except Exception as e:
                if self.verbose:
                    print(f"AI detector initialization failed: {e}")
    
    def _setup_ai_detector(self):
        """Set up the AI content detector model (optional)."""
        model_name = "roberta-base-openai-detector"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("AI detector initialized")
        return {"model": model, "tokenizer": tokenizer}
    
    def _load_document(self, file_path: str) -> List[str]:
        """Load and split a document into chunks."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        else:
            # Assume it's a text file
            loader = TextLoader(str(file_path))
        
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        return text_splitter.split_documents(documents)
    
    def _check_submission_relevance(self, submission_text: str, assignment_details: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if the submission is relevant to the assignment."""
        template = """
        You are an academic evaluator tasked with determining if a student submission is relevant to the assigned topic.
        
        # Assignment Details
        Assignment Type: {assignment_type}
        Subject: {subject}
        Title: {title}
        Description: {description}
        
        # Instructions to Students
        {instructions}
        
        # Student Submission (excerpt)
        {submission_excerpt}
        
        # Task
        Determine if this submission is addressing the assignment topic and following instructions.
        Consider:
        1. Does the submission address the core topic of the assignment?
        2. Is the submission attempting to follow the assignment instructions?
        3. Does the submission appear to be for a completely different assignment?
        
        Provide your assessment as:
        - RELEVANT: If the submission appears to be addressing this assignment
        - NOT_RELEVANT: If the submission appears to be for a completely different assignment
        
        First analyze the submission in relation to the assignment, then provide your final assessment.
        """
        
        # Take first ~1000 characters as an excerpt for relevance check
        submission_excerpt = submission_text[:1000] + ("..." if len(submission_text) > 1000 else "")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are an academic evaluator assessing assignment relevance."),
            HumanMessagePromptTemplate.from_template(template)
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        if self.verbose and self.console:
            with self.console.status("[bold green]Checking submission relevance..."):
                response = chain.run(
                    assignment_type=assignment_details.get("assignment_type", ""),
                    subject=assignment_details.get("subject", ""),
                    title=assignment_details.get("title", ""),
                    description=assignment_details.get("description", ""),
                    instructions=assignment_details.get("instructions", ""),
                    submission_excerpt=submission_excerpt
                )
        else:
            response = chain.run(
                assignment_type=assignment_details.get("assignment_type", ""),
                subject=assignment_details.get("subject", ""),
                title=assignment_details.get("title", ""),
                description=assignment_details.get("description", ""),
                instructions=assignment_details.get("instructions", ""),
                submission_excerpt=submission_excerpt
            )
        
        is_relevant = "NOT_RELEVANT" not in response.upper()
        return is_relevant, response
    
    def _extract_rubric_criteria(self, rubric_text: str) -> List[Dict[str, Any]]:
        """Extract and structure rubric criteria from rubric text."""
        template = """
        Extract grading criteria from the following rubric:
        
        {rubric_text}
        
        Return a JSON array of criteria objects, each with:
        1. "name": The name or title of the criterion
        2. "description": Description of what's being evaluated
        3. "points": Maximum points possible (numeric)
        
        Format as valid JSON without explanations.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are an AI assistant that extracts structured data from text."),
            HumanMessagePromptTemplate.from_template(template)
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt, output_key="criteria")
        
        if self.verbose and self.console:
            with self.console.status("[bold green]Analyzing rubric..."):
                response = chain.run(rubric_text=rubric_text)
        else:
            response = chain.run(rubric_text=rubric_text)
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                criteria = json.loads(json_match.group(0))
                return criteria
            except json.JSONDecodeError:
                pass
        
        # Fallback
        return [{"name": "Overall Quality", "description": "General quality of submission", "points": 100}]
    
    def _evaluate_submission_chunk(self, chunk: str, assignment_details: Dict[str, Any], rubric_criteria: List[Dict[str, Any]]) -> str:
        """Evaluate a chunk of the submission against the rubric."""
        template = """
        # Assignment Context
        Assignment Type: {assignment_type}
        Subject: {subject}
        Title: "{title}"
        
        # Assignment Instructions
        {instructions}
        
        # Relevant Content from Student Submission (Part {chunk_num})
        {chunk_content}
        
        # Task
        Analyze this portion of the student's work against the following criteria:
        {criteria_formatted}
        
        For this specific portion of content, provide observations related to each criterion.
        Note strengths, weaknesses, and specific examples from the text.
        These notes will be combined with analysis of other sections to create the final evaluation.
        
        Focus only on what's present in this chunk. Note if this section is particularly strong or weak in any criteria.
        """
        
        criteria_formatted = "\n".join([f"- {c['name']}: {c['description']} (Max: {c['points']} points)" for c in rubric_criteria])
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are an educational assessment expert evaluating student work."),
            HumanMessagePromptTemplate.from_template(template)
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        return chain.run(
            assignment_type=assignment_details.get("assignment_type", ""),
            subject=assignment_details.get("subject", ""),
            title=assignment_details.get("title", ""),
            instructions=assignment_details.get("instructions", ""),
            chunk_content=chunk,
            chunk_num="X",  # This will be different for each chunk
            criteria_formatted=criteria_formatted
        )
    
    def _combine_evaluations(self, evaluations: List[str], assignment_details: Dict[str, Any], rubric_criteria: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine chunk evaluations into a final assessment."""
        # Load JSON response format template
        template_path = Path(__file__).parent / "response_format.json"
        try:
            with open(template_path, "r") as f:
                response_format_template = f.read()
        except Exception:
            response_format_template = ""
        
        template = """
        # Assignment Context
        Assignment Type: {assignment_type}
        Subject: {subject}
        Title: "{title}"
        
        # Assignment Instructions
        {instructions}
        
        # Rubric Criteria
        {criteria_formatted}
        
        # Combined Analysis from All Sections
        {combined_analysis}
        
        # Task
        Based on the above analysis of all sections of the student submission, create a comprehensive evaluation that includes:
        
        1. Grading for each rubric criterion with specific comments
        2. Overall grade with justification
        3. Key strengths of the submission
        4. Areas for improvement
        5. Specific action items for the student
        
        Format your response using appropriate markdown headings and structure for clarity.
        """
        
        criteria_formatted = "\n".join([f"- {c['name']}: {c['description']} (Max: {c['points']} points)" for c in rubric_criteria])
        combined_analysis = "\n\n".join([f"SECTION ANALYSIS:\n{eval}" for eval in evaluations])
        
        # Escape curly braces in JSON schema before formatting
        escaped_schema = response_format_template.replace('{', '{{').replace('}', '}}')
        template += (
            "\nPlease strictly output a JSON object matching the following template:\n"
            f"{escaped_schema}\n"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are an educational assessment expert delivering a comprehensive evaluation of student work."),
            HumanMessagePromptTemplate.from_template(template)
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        if self.verbose and self.console:
            with self.console.status("[bold green]Generating final evaluation..."):
                response = chain.run(
                    assignment_type=assignment_details.get("assignment_type", ""),
                    subject=assignment_details.get("subject", ""),
                    title=assignment_details.get("title", ""),
                    instructions=assignment_details.get("instructions", ""),
                    criteria_formatted=criteria_formatted,
                    combined_analysis=combined_analysis
                )
        else:
            response = chain.run(
                assignment_type=assignment_details.get("assignment_type", ""),
                subject=assignment_details.get("subject", ""),
                title=assignment_details.get("title", ""),
                instructions=assignment_details.get("instructions", ""),
                criteria_formatted=criteria_formatted,
                combined_analysis=combined_analysis
            )
        
        # # Save response to file
        print(response)
        # with open(f"{_output_path}/response.txt", "w") as f:
        #     f.write(response)

        # Parse JSON response
        response = response.replace("```json\n", "").replace("```", "")
        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            parsed_response = {"error": "JSON decode error", "raw_response": response}

        return parsed_response
    
    def _check_for_ai_content(self, text: str) -> Dict[str, Any]:
        """Check if content appears to be AI-generated."""
        if not self.ai_detector:
            return {"performed": False, "message": "AI content detection not available"}
        
        # Split text into smaller parts if it's too long
        max_length = self.ai_detector["tokenizer"].model_max_length
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        results = []
        sections = self._split_into_sections(text)
        for section in sections:
            try:
                inputs = self.ai_detector["tokenizer"](section, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.ai_detector["model"](**inputs)
                
                # Get scores (0 = human, 1 = AI)
                scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
                results.append({
                    "section": section,
                    "human_score": scores[0],
                    "ai_score": scores[1]
                    
                })
            except Exception as e:
                if self.verbose:
                    print(f"Error in AI detection: {e}")
                continue
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into sections (e.g. paragraphs)."""
        # Split on double newline
        sections = text.split("\n\n")
        
        # Split on headings (e.g. # Heading)
        new_sections = []
        for section in sections:
            headings = re.findall(r"^#+\s+.*", section, re.MULTILINE)
            if headings:
                new_sections.extend(headings + [section])
            else:
                new_sections.append(section)
        
        return new_sections
        
        # Calculate averages
        if not results:
            return {"performed": False, "message": "AI detection failed"}
        
        avg_human_score = sum(r["human_score"] for r in results) / len(results) * 100
        avg_ai_score = sum(r["ai_score"] for r in results) / len(results) * 100
        
        return {
            "performed": True,
            "human_score": avg_human_score,
            "ai_score": avg_ai_score,
            "assessment": "Likely human-written" if avg_human_score > avg_ai_score else "Likely AI-generated",
            "confidence": max(avg_human_score, avg_ai_score)
        }
    
    def _check_for_plagiarism(self, text: str, subject: str) -> Dict[str, Any]:
        """Simulated plagiarism check using LLM pattern recognition."""
        template = """
        # Student Submission (Excerpt)
        {text}
        
        # Task
        You are an expert in detecting plagiarism. Analyze this submission for:
        1. Unusual shifts in writing style or tone
        2. Text that appears unusually polished or academic compared to surrounding text
        3. Passages that use specialized knowledge without proper attribution
        4. Content that seems like it may be directly quoted without citation
        
        This is a {subject} assignment. Consider the expected knowledge level of a student in this subject.
        
        If you identify potential plagiarism, explain why the passage raises concerns.
        If you find no evidence of potential plagiarism, state that no obvious indicators were found.
        
        Provide your assessment in a structured format with specific examples if plagiarism is suspected.
        I am expecting the output in below json format
        {"plagiarism_score_in_percent": 0,
            "matched_content": [
                {
                    "submission_chunk": "",
                    "match_percentage": 0,
                    "source": ""
            }
        ]
        }
        """
        
        # Use a smaller chunk to keep the request manageable
        text_sample = text[:3000] + ("..." if len(text) > 3000 else "")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are an academic integrity expert."),
            HumanMessagePromptTemplate.from_template(template)
        ])
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        if self.verbose and self.console:
            with self.console.status("[bold green]Checking for potential plagiarism..."):
                response = chain.run(text=text_sample, subject=subject)
        else:
            response = chain.run(text=text_sample, subject=subject)
        
        # Determine if plagiarism was detected based on response content
        plagiarism_detected = "no obvious indicators" not in response.lower() and \
                             ("potential plagiarism" in response.lower() or 
                              "plagiarism is suspected" in response.lower())
        
        return {
            "performed": True,
            "plagiarism_detected": plagiarism_detected,
            "analysis": response
        }
    
    def create_evaluation_workflow(self):
        """Create a LangGraph workflow for the evaluation process."""
        
        # Define the state
        class State(dict):
            assignment_details: Dict[str, Any]
            submission_chunks: List[str]
            is_relevant: bool
            relevance_analysis: str
            rubric_criteria: List[Dict[str, Any]]
            chunk_evaluations: List[str]
            final_evaluation: str
            ai_content_check: Dict[str, Any]
            plagiarism_check: Dict[str, Any]
        
        # Define workflow nodes
        def check_relevance(state: State) -> State:
            """Check if submission is relevant to assignment."""
            submission_text = "".join([chunk.page_content for chunk in state["submission_chunks"]])
            is_relevant, relevance_analysis = self._check_submission_relevance(
                submission_text, state["assignment_details"]
            )
            state["is_relevant"] = is_relevant
            state["relevance_analysis"] = relevance_analysis
            return state
        
        def extract_rubric(state: State) -> State:
            """Extract structured rubric criteria."""
            rubric_text = state["assignment_details"].get("rubric", "")
            state["rubric_criteria"] = self._extract_rubric_criteria(rubric_text)
            return state
        
        def evaluate_chunks(state: State) -> State:
            """Evaluate each submission chunk."""
            chunk_evaluations = []
            
            for i, chunk in enumerate(state["submission_chunks"]):
                if self.verbose:
                    print(f"Evaluating chunk {i+1}/{len(state['submission_chunks'])}...")
                evaluation = self._evaluate_submission_chunk(
                    chunk.page_content, 
                    state["assignment_details"],
                    state["rubric_criteria"]
                )
                chunk_evaluations.append(evaluation)
            
            state["chunk_evaluations"] = chunk_evaluations
            return state
        
        def create_final_evaluation(state: State) -> State:
            """Combine evaluations into final assessment."""
            state["final_evaluation"] = self._combine_evaluations(
                state["chunk_evaluations"],
                state["assignment_details"],
                state["rubric_criteria"]
            )
            return state
        
        def check_ai_content(state: State) -> State:
            """Check for AI-generated content."""
            submission_text = "".join([chunk.page_content for chunk in state["submission_chunks"]])
            state["ai_content_check"] = self._check_for_ai_content(submission_text)
            return state
        
        def check_plagiarism(state: State) -> State:
            """Check for potential plagiarism."""
            submission_text = "".join([chunk.page_content for chunk in state["submission_chunks"]])
            subject = state["assignment_details"].get("subject", "")
            state["plagiarism_check"] = self._check_for_plagiarism(submission_text, subject)
            return state
        
        # Create workflow graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("check_relevance", check_relevance)
        workflow.add_node("extract_rubric", extract_rubric)
        workflow.add_node("evaluate_chunks", evaluate_chunks)
        workflow.add_node("create_final_evaluation", create_final_evaluation)
        workflow.add_node("check_ai_content", check_ai_content)
        workflow.add_node("check_plagiarism", check_plagiarism)
        
        # Add edges
        workflow.add_edge(START, "check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            lambda state: "extract_rubric" if state["is_relevant"] else END,
            {
                "extract_rubric": "extract_rubric", 
                END: END
            }
        )
        workflow.add_edge("extract_rubric", "evaluate_chunks")
        workflow.add_edge("evaluate_chunks", "create_final_evaluation")
        workflow.add_edge("create_final_evaluation", END)
        # workflow.add_edge("check_ai_content", "check_plagiarism")
        # workflow.add_edge("check_plagiarism", END)
        
        # Compile
        return workflow.compile()
    
    def evaluate_submission(
        self, 
        submission_path: str, 
        assignment_type: str,
        subject: str,
        title: str,
        description: str,
        instructions: str,
        rubric: str,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a student submission against assignment details and rubric.
        
        Args:
            submission_path: Path to the submission file (PDF or text)
            assignment_type: Type of assignment (essay, research paper, etc.)
            subject: Subject of the assignment
            title: Title of the assignment
            description: Description of the assignment
            instructions: Instructions given to students
            rubric: Grading rubric text
            output_path: Optional path to save evaluation report
            
        Returns:
            Dict containing evaluation results
        """
        if self.verbose and self.console:
            self.console.print(f"[bold]Evaluating submission:[/bold] {submission_path}")
            self.console.print(f"[bold]Assignment:[/bold] {title} ({assignment_type})")
        
        # Load document
        try:
            # Splits the document in the chunks
            submission_chunks = self._load_document(submission_path)
        except Exception as e:
            error_msg = f"Error loading submission: {str(e)}"
            if self.verbose:
                print(error_msg)
            return {"success": False, "error": error_msg}
        
        # Prepare assignment details
        assignment_details = {
            "assignment_type": assignment_type,
            "subject": subject,
            "title": title,
            "description": description,
            "instructions": instructions,
            "rubric": rubric
        }
        
        # Create initial state
        initial_state = {
            "assignment_details": assignment_details,
            "submission_chunks": submission_chunks,
            "is_relevant": False,
            "relevance_analysis": "",
            "rubric_criteria": [],
            "chunk_evaluations": [],
            "final_evaluation": "",
            "ai_content_check": {},
            "plagiarism_check": {}
        }
        
        # Create and run workflow
        workflow = self.create_evaluation_workflow()
        final_state = workflow.invoke(initial_state)

        # Save final state to file
        # with open(f"{_output_path}/final_state.txt", "w") as f:
        #     f.write(str(final_state))

        # print(json.loads(final_state, indent=4))
        # Prepare results
        results = {
            "success": True,
            "is_relevant": final_state.get("is_relevant", False),
            "relevance_analysis": final_state.get("relevance_analysis", ""),
            "evaluation": final_state.get("final_evaluation", "").get("evaluation", ""),
            # "ai_content_check": final_state.get("ai_content_check", {"performed": False}),
            # "plagiarism_check": final_state.get("plagiarism_check", {"performed": False})
        }

        # Save results to file
        # with open(f"{_output_path}/results.txt", "w") as f:
        #     f.write(json.dumps(results, indent=4))
        # print(results)
        # print(type(results))

        # Save report if output path provided
        if output_path and results["success"]:
            try:
                with open(output_path, "w") as f:
                    # Create markdown report
                    report = f"# Evaluation Report: {title}\n\n"
                    report += f"**Assignment Type:** {assignment_type}  \n"
                    report += f"**Subject:** {subject}\n\n"
                    
                    # Add evaluation
                    report += results["evaluation"]
                    
                    # # Add AI content check if performed
                    # if results["ai_content_check"].get("performed", False):
                    #     report += "\n\n## AI Content Assessment\n\n"
                    #     report += f"**Assessment:** {results['ai_content_check'].get('assessment', 'N/A')}\n\n"
                    #     report += f"**Human-written score:** {results['ai_content_check'].get('human_score', 0):.2%}  \n"
                    #     report += f"**AI-generated score:** {results['ai_content_check'].get('ai_score', 0):.2%}\n\n"
                    
                    # # Add plagiarism check if performed
                    # if results["plagiarism_check"].get("performed", False):
                    #     report += "\n\n## Plagiarism Assessment\n\n"
                    #     report += results["plagiarism_check"].get("analysis", "No analysis available.")
                    
                    f.write(report)
                
                if self.verbose:
                    print(f"Evaluation report saved to {output_path}")
                
                results["report_path"] = output_path
            except Exception as e:
                if self.verbose:
                    print(f"Error saving report: {e}")
        
        # Display results if verbose
        if self.verbose and self.console and results["success"]:
            if not results["is_relevant"]:
                self.console.print(Panel(
                    "[bold red]SUBMISSION REJECTED: Not relevant to assignment.[/bold red]\n\n" + 
                    results["relevance_analysis"],
                    title="Relevance Check Failed",
                    border_style="red"
                ))
            else:
                self.console.print(Markdown(str(results["evaluation"])))
                
                # if results["ai_content_check"].get("performed", False):
                #     human_score = results["ai_content_check"].get("human_score", 0) * 100
                #     ai_score = results["ai_content_check"].get("ai_score", 0) * 100
                    
                #     color = "green" if human_score > ai_score else "yellow"
                #     self.console.print(Panel(
                #         f"[bold]{results['ai_content_check'].get('assessment', 'N/A')}[/bold]\n\n" +
                #         f"Human-written score: {human_score:.1f}%\n" +
                #         f"AI-generated score: {ai_score:.1f}%",
                #         title="AI Content Detection",
                #         border_style=color
                #     ))
                
                # if results["plagiarism_check"].get("performed", False):
                #     detected = results["plagiarism_check"].get("plagiarism_detected", False)
                #     color = "yellow" if detected else "green"
                #     title = "Potential Plagiarism Detected" if detected else "No Plagiarism Detected"
                    
                #     self.console.print(Panel(
                #         results["plagiarism_check"].get("analysis", "No analysis available."),
                #         title=title,
                #         border_style=color
                #     ))
        
        return results

def main():
    parser = argparse.ArgumentParser(description="AI Assignment Evaluator")
    parser.add_argument("--submission", required=True, help="Path to submission file (PDF or text)")
    parser.add_argument("--assignment-type", required=True, help="Type of assignment (essay, research paper, etc.)")
    parser.add_argument("--subject", required=True, help="Subject of the assignment")
    parser.add_argument("--title", required=True, help="Title of the assignment")
    parser.add_argument("--description", required=True, help="Description of the assignment")
    parser.add_argument("--instructions", required=True, help="Instructions given to students")
    parser.add_argument("--rubric", required=True, help="Path to rubric text file")
    parser.add_argument("--output", help="Path to save evaluation report (optional)")
    parser.add_argument("--api-key", help="OpenAI API key (can also use OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    # python .\ai_evaluator.py   --submission '.\student_essay old.pdf'  --assignment-type 'essay' --subject 'environment' --title 'Climate Change Impact Analysis' --description 'write essay on the climate change Impact analysis' --instructions .\instructions.txt --rubric .\rubric.txt 
    args = parser.parse_args()
    
    # Read rubric file
    try:
        with open(args.rubric, "r") as f:
            rubric_text = f.read()
    except Exception as e:
        print(f"Error reading rubric file: {e}")
        return 1
    
    try:
        # Initialize evaluator
        evaluator = AIAssignmentEvaluator(
            api_key=args.api_key,
            model=args.model,
            verbose=args.verbose
        )
        
        # Run evaluation
        results = evaluator.evaluate_submission(
            submission_path=args.submission,
            assignment_type=args.assignment_type,
            subject=args.subject,
            title=args.title,
            description=args.description,
            instructions=args.instructions,
            rubric=rubric_text,
            output_path=args.output
        )
        
        if not results["success"]:
            print(f"Evaluation failed: {results.get('error', 'Unknown error')}")
            return 1
            
        elif not results["is_relevant"]:
            print("Submission rejected: Not relevant to assignment.")
            if args.verbose:
                print(results["relevance_analysis"])
            return 0
            
        elif not args.verbose:
            print("Evaluation completed successfully!")
            if args.output:
                print(f"Report saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())