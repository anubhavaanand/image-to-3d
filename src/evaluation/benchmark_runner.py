import pandas as pd
from typing import List, Dict, Any
from ..engines.base import AbstractEngine
import time
from PIL import Image
import numpy as np

class BenchmarkRunner:
    def __init__(self, dataset_path: str, engines: List[AbstractEngine], metrics: List[str]):
        self.dataset_path = dataset_path
        self.engines = engines
        self.metrics = metrics
        self.results = []
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        # Mock dataset loading
        return [{"id": f"sample_{i}", "image": Image.new("RGB", (256, 256), color="white")} for i in range(5)]
        
    def run(self):
        samples = self.load_dataset()
        for engine in self.engines:
            engine_info = engine.get_engine_info()
            engine_name = engine_info["name"]
            
            for sample in samples:
                start_time = time.time()
                try:
                    result = engine.predict(sample["image"])
                    latency = time.time() - start_time
                    
                    # Mock metric evaluation
                    metrics_result = {
                        "chamfer_distance": np.random.rand(),
                        "f_score_0.1": np.random.rand(),
                        "psnr": np.random.uniform(15, 30),
                        "lpips": np.random.uniform(0.0, 0.5)
                    }
                    
                    self.results.append({
                        "engine": engine_name,
                        "sample_id": sample["id"],
                        "latency_s": latency,
                        **metrics_result
                    })
                except Exception as e:
                    print(f"Error evaluating {engine_name} on {sample['id']}: {e}")
                    
    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)
        
    def export_latex_table(self) -> str:
        df = self.get_results_df()
        if df.empty:
            return ""
            
        summary = df.groupby("engine").mean(numeric_only=True).round(3)
        return summary.to_latex(
            caption="Evaluation results of Image-to-3D models.",
            label="tab:evaluation_results",
            index=True
        )
