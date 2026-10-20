import os

import streamlit as st

# set SHOW_ANSWERS=1 to render all quiz answers (used for automated rendering checks)
_SHOW_ALL = os.environ.get("SHOW_ANSWERS") == "1"


def takeaways(points):
    """Short summary box shown at the end of a page."""
    st.info("**💡 Key takeaways**\n\n" + "\n".join(f"- {p}" for p in points))


def test_yourself(key, items):
    """Exam-style self-test: each question with an answer that can be revealed via a toggle."""
    with st.expander("📝 Test yourself — exam-style questions"):
        for i, (question, answer) in enumerate(items, start=1):
            st.markdown(f"**Q{i}.** {question}")
            if st.toggle("Show answer", key=f"{key}_q{i}", value=_SHOW_ALL):
                st.markdown(answer)
            if i < len(items):
                st.markdown("---")
