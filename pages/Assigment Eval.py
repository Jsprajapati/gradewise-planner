# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import tempfile
import os
from grading_logic import AIAssignmentEvaluator
st.set_page_config(page_title="Assignment Evaluation", page_icon="") # Specific config for this page

st.title("Assignment Evaluation")

with st.form("assignment_eval_form"):
    assignment_type = st.selectbox("Assignment type", ["Essay", "Research Paper", "Project", "Presentation", "Quiz", "Exam", "Discussion"])

    subject = st.text_input("Subject")
    title_of_assignment = st.text_input("Title of assignment")
    instruction = st.text_area("Instruction")
    description = st.text_area("Description")
    rubric = st.text_area("Rubric")
    submission_doc = st.file_uploader("Submission Document")

    # Every form must have a submit button
    submitted = st.form_submit_button("Evaluate")
    result = []
    if submitted:
        with st.status("Evaluating...", expanded=True) as status:
            evaluator = AIAssignmentEvaluator(
                api_key=st.session_state.api_key,
                model=st.session_state.model
            )
            # props of object?
            # st.write(submission_doc)
            temp_dir = tempfile.mkdtemp()
            submission_file_path = os.path.join(temp_dir, submission_doc.name)
            # Save the uploaded file to the temporary path
            try:
                with open(submission_file_path, "wb") as f:
                    f.write(submission_doc.getvalue())
                # st.success(f"File saved temporarily to: `{submission_file_path}`")
            except Exception as e:
                st.error(f"Error saving file: {e}")
                submission_file_path = None # Reset path if save failed

            # Run evaluation
            results = evaluator.evaluate_submission(
                submission_path=submission_file_path,
                assignment_type=assignment_type,
                subject=subject,
                title=title_of_assignment,
                description=description,
                instructions=instruction,
                rubric=rubric
            )
        
            if not results["success"]:
                status.update(label=f"Evaluation failed: {results.get('error', 'Unknown error')}", state="error", expanded=False)
                
            elif not results["is_relevant"]:
                status.update(label="Submission rejected: Not relevant to assignment.", state="error", expanded=False)
            else:
                status.update(label="Evaluation completed successfully!", state="complete", expanded=False)
                st.markdown("##### Overall Grade")
                st.metric("Score",results["evaluation"]["overall_grade"]["score"])
                st.markdown(""+results["evaluation"]["overall_grade"]["justification"])
                with st.expander("Evaluation"):
                    st.json(results["evaluation"])

                with st.expander("Rubric Grades"):
                    rubric_criteria = []
                    rubric_grades = []
                    rubric_text = []
                    rubric_grades = results["evaluation"].get("rubric_grades", []) or results["evaluation"].get("grading", [])
                    for key, value in rubric_grades.items():
                        rubric_criteria.append(key)
                        rubric_grades.append(value["score"])
                        rubric_text.append(value["comments"])
                    rubric_analysis = pd.DataFrame({"Criteria": rubric_criteria, "Grade": rubric_grades, "Comments": rubric_text}, index=range(1, len(rubric_criteria) + 1))
                    st.dataframe(rubric_analysis, use_container_width=True)

                with st.expander("Strengths"):
                    strengths = results["evaluation"].get("strengths", []) or results["evaluation"].get("key_strengths", [])
                    if strengths:
                        for strength in strengths:
                            st.markdown(f"- {strength}")

                with st.expander("Areas for Improvement"):
                    areas_for_improvement = results["evaluation"].get("areas_for_improvement", [])
                    if areas_for_improvement:
                        for area in areas_for_improvement:
                            st.markdown(f"- {area}")

                with st.expander("Action Items"):
                    action_items = results["evaluation"].get("action_items", [])
                    if action_items:
                        for action_item in action_items:
                            st.markdown(f"- {action_item}")
            



