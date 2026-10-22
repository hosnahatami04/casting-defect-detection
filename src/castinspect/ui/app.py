"""Streamlit "inspection station". Run with:
streamlit run src/castinspect/ui/app.py

Reads the ONNX model and frozen policy, shows the three-way decision and the
defect probability for an uploaded or sample image, and a threshold slider
whose effect on escapes and false rejects is read off the out-of-fold
predictions. A second page shows the evaluation figures.
"""

from pathlib import Path

import streamlit as st
from PIL import Image

from castinspect.decision.calibration import apply_temperature, probabilities_to_logits
from castinspect.decision.policy import DecisionPolicy, PolicyConfig, load_policy
from castinspect.deploy.runtime import OnnxInspector
from castinspect.ui.logic import load_oof_scores, read_results, slider_effect

ARTIFACTS = Path("artifacts")
REPORTS = Path("reports")
ONNX_PATH = ARTIFACTS / "onnx" / "efficientnet_b0.onnx"
POLICY_PATH = Path("configs/policy.yaml")


@st.cache_resource
def _load() -> tuple[OnnxInspector, PolicyConfig, float]:
    policy_config = load_policy(POLICY_PATH)
    inspector = OnnxInspector(ONNX_PATH)
    return inspector, policy_config, policy_config.temperature


def _calibrated_probability(
    inspector: OnnxInspector, image: Image.Image, temperature: float
) -> float:
    raw = inspector.predict_one(image)
    return float(apply_temperature(probabilities_to_logits([raw]), temperature)[0])


def inspection_page() -> None:
    st.title("Casting inspection station")
    inspector, policy_config, temperature = _load()
    uploaded = st.file_uploader("Upload a part image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload a top-view casting image to inspect it.")
        return
    image = Image.open(uploaded).convert("L")
    st.image(image, caption="input", width=320)
    probability = _calibrated_probability(inspector, image, temperature)

    reject = st.slider(
        "Reject threshold",
        0.0,
        1.0,
        float(policy_config.thresholds.reject),
        0.01,
    )
    policy = DecisionPolicy(
        review_threshold=min(policy_config.thresholds.review, reject),
        reject_threshold=reject,
        version=policy_config.version,
    )
    decision = policy.decide(probability)
    colour = {"pass": "green", "reject": "red", "manual_review": "orange"}[decision.action]
    st.markdown(f"### :{colour}[{decision.action.upper()}]")
    st.write(f"defect probability: {probability:.4f}")

    oof = ARTIFACTS / "oof" / f"{policy_config.model}.parquet"
    if oof.is_file():
        labels, probabilities = load_oof_scores(oof)
        calibrated = apply_temperature(probabilities_to_logits(probabilities), temperature)
        effect = slider_effect(labels, calibrated, policy_config.thresholds.review, reject)
        st.write(
            f"At this threshold on held-out data: {effect.escapes} escapes "
            f"({effect.escape_rate:.2%}), {effect.false_rejects} false rejects "
            f"({effect.false_reject_rate:.2%}), review share {effect.review_share:.2%}."
        )


def figures_page() -> None:
    st.title("Evaluation figures")
    results = read_results(REPORTS / "results.json")
    if results:
        st.json(results)
    for name in ("data_efficiency.png", "robustness_oof.png", "reliability.png"):
        figure = REPORTS / "figures" / name
        if figure.is_file():
            st.image(str(figure), caption=name)


def main() -> None:
    st.set_page_config(page_title="castinspect", layout="centered")
    page = st.sidebar.radio("Page", ["Inspection", "Figures"])
    if page == "Inspection":
        inspection_page()
    else:
        figures_page()


if __name__ == "__main__":
    main()
