import streamlit as st
import json
import re
import torch
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import tempfile
import os

st.set_page_config(page_title="LogGPT Anomaly Detector", layout="wide")

# Load model + tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("./final_model")
    model = GPT2LMHeadModel.from_pretrained("./final_model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Load threshold
with open("threshold_FinalModel.json") as f:
    threshold = json.load(f).get("threshold", 1.0)

# Drain3
@st.cache_resource
def load_template_miner():
    config = TemplateMinerConfig()
    config.load("drain3.ini")
    return TemplateMiner(config=config)

template_miner = load_template_miner()

# Regexes
timestamp_regex = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})(?:,(\d{3}))?")
level_regex = re.compile(r"-(INFO|DEBUG|WARN|ERROR|TRACE|FATAL)\b")

# Score log
def score_log(log_text, tokenizer, model, device, max_length=512):
    inputs = tokenizer(
        log_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()

# UI
st.title("🔍 LogGPT Anomaly Detector")
st.markdown("Upload your `.log` file below. The model will parse and detect anomalies.")

uploaded_file = st.file_uploader("Upload log file", type=["log"])

if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    in_log_path = os.path.join(temp_dir, "input.log")
    out_jsonl_path = os.path.join(temp_dir, "structured.jsonl")
    out_anomalies_path = os.path.join(temp_dir, "anomalies.txt")

    with open(in_log_path, "wb") as f:
        f.write(uploaded_file.read())

    with open(in_log_path, "r") as infile, \
         open(out_jsonl_path, "w") as outfile, \
         open(out_anomalies_path, "w") as anomaly_file:

        metadata_lines = []
        log_blocks = []
        current_block = []
        in_metadata = True

        for line in infile:
            stripped = line.rstrip()

            if timestamp_regex.match(stripped):
                in_metadata = False
                if current_block:
                    log_blocks.append("\n".join(current_block))
                    current_block = []
            elif in_metadata:
                metadata_lines.append(stripped)
                continue

            current_block.append(stripped)

        if current_block:
            log_blocks.append("\n".join(current_block))

        # For simplicity, skip metadata parsing in UI
        parsed_metadata = {}

        anomaly_count = 0
        st.markdown("### 🚨 Anomalous Logs Detected")
        for log_block in log_blocks:
            log_message = log_block.strip()
            first_line = log_message.splitlines()[0]

            ts_match = timestamp_regex.match(first_line)
            if ts_match:
                date_str, time_str, millis_str = ts_match.groups()
                if millis_str:
                    dt = datetime.strptime(f"{date_str} {time_str},{millis_str}", "%Y-%m-%d %H:%M:%S,%f")
                else:
                    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                timestamp_iso = dt.isoformat()
                date_only = dt.date().isoformat()
            else:
                timestamp_iso, date_only = None, None

            result = template_miner.add_log_message(log_message)
            template = result["template_mined"]
            params = template_miner.extract_parameters(template, log_message)
            level = level_regex.search(log_message)

            structured = {
                "type": "log",
                "timestamp": timestamp_iso,
                "date": date_only,
                "level": level.group(1) if level else None,
                "raw": log_message,
                "template": template,
                "params": [p.value for p in params] if params else [],
                "metadata": parsed_metadata,
                "change_type": result.get("change_type"),
                "cluster_id": result.get("cluster_id"),
                "cluster_size": result.get("cluster_size"),
                "cluster_count": result.get("cluster_count"),
            }

            outfile.write(json.dumps(structured) + "\n")

            log_text = (
                f"Timestamp: {structured['timestamp']}, Date: {structured['date']}, "
                f"Level: {structured['level']}\n"
                f"Raw: {structured['raw']}\n"
                f"Template: {structured['template']}, Params: {structured['params']}\n"
                f"Change Type: {structured['change_type']}, Cluster ID: {structured['cluster_id']}\n"
            )

            score = score_log(log_text, tokenizer, model, device)
            if score > threshold:
                st.code(log_text + f"\nAnomaly Score: {score:.4f}", language="text")
                anomaly_file.write(json.dumps(structured) + "\n\n")
                anomaly_count += 1

        st.success(f"✅ Detected {anomaly_count} anomalous logs.")

        # Provide download link
        with open(out_anomalies_path, "rb") as f:
            st.download_button(
                label="📥 Download Anomalous Logs",
                data=f,
                file_name="anomalous_logs.txt",
                mime="text/plain"
            )
