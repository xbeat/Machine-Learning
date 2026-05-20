#!/usr/bin/env python3
"""Organize markdown files into topic-based folders."""

import os
import shutil
import re

BASE = "/home/user/Machine-Learning"

CATEGORIES = {
    "LLMs-and-GenAI": [
        r"llm", r"large language model", r"rag", r"retrieval.augmented", r"langchain",
        r"langgraph", r"prompt engineer", r"chatgpt", r"openai", r"gemini", r"mistral",
        r"llama", r"gpt", r"agentic", r"agent.*rag", r"rag.*agent", r"vector database",
        r"vector db", r"fine.tun.*llm", r"llm.*fine.tun", r"foundation model",
        r"hallucination", r"chunking.*rag", r"rag.*chunk", r"auto.document retrieval",
        r"bond.*llm", r"accelerated generation", r"token.*llm", r"context window",
        r"instruction tun", r"rlhf", r"direct preference", r"chain.of.thought",
        r"few.shot", r"zero.shot.*llm", r"in.context learn", r"text.to.sql.*llm",
        r"vanna.*sql", r"knowledge graph.*llm", r"multimodal.*llm", r"llm.*multimodal",
        r"mplug", r"ai.*agent", r"automated.*agent", r"agentic rag",
        r"applications of llm", r"addressing.*llm", r"accelerating.*llm",
    ],
    "NLP-and-Transformers": [
        r"nlp", r"natural language process", r"bert", r"bart", r"\bt5\b", r"transformer",
        r"tokeniz", r"wordpiece", r"word2vec", r"word embed", r"sentence embed",
        r"sentiment analysis", r"named entity", r"text classif", r"sequence.to.sequence",
        r"seq2seq", r"language model(?!.*large)", r"text generation", r"speech",
        r"bag of words", r"tfidf", r"tf.idf", r"spacy", r"nltk", r"aspect.based",
        r"zero.shot classif", r"zero.shot learn", r"automatic.*domain adapt",
        r"attention mechanism", r"attention is all", r"fascinating transformer",
        r"why transformer", r"word tokeniz", r"text chunk", r"using.*regex.*nlp",
    ],
    "Computer-Vision": [
        r"computer vision", r"image classif", r"image recogni", r"image process",
        r"image segment", r"object detect", r"face.*generat", r"face.*model",
        r"active shape model", r"grad.cam", r"cnn.*visualiz", r"visualiz.*cnn",
        r"convolutional neural", r"resnet", r"vgg", r"yolo", r"zfnet", r"vit\b",
        r"vision transformer", r"ecg.*cnn", r"transfer learn.*cnn", r"3d packing.*depth",
        r"monocular depth", r"object.*track", r"image.*generat", r"3d graphics",
        r"rendering.*python", r"cifar", r"imagenet", r"alexnet", r"inception",
        r"batch norm.*cnn", r"visualiz.*batch norm.*cnn",
    ],
    "Reinforcement-Learning": [
        r"reinforcement learn", r"q.learning", r"policy gradient", r"reward", r"environment.*rl",
        r"markov decision", r"multi.agent.*rl", r"deep.q", r"actor.critic",
        r"proximal policy", r"ppo\b", r"dqn\b", r"temporal difference", r"bellman",
    ],
    "Deep-Learning": [
        r"neural network", r"deep learning", r"backprop", r"gradient descent",
        r"activation function", r"dropout", r"batch normalization", r"batch norm",
        r"autoencoder", r"\bgan\b", r"generative adversarial", r"variational autoencoder",
        r"\bvae\b", r"diffusion model", r"weight init", r"vanishing.*gradient",
        r"exploding.*gradient", r"learning rate", r"epoch", r"perceptron",
        r"multilayer", r"recurrent neural", r"\brnn\b", r"\blstm\b", r"\bgru\b",
        r"encoder.*decoder", r"u.net", r"residual network", r"skip connection",
        r"layer normalization", r"mixed precision", r"quantization.*neural",
        r"pruning.*neural", r"knowledge distill", r"neural.*compress",
        r"adversarial attack", r"adversarial robust", r"adversarial example",
        r"kolmogorov.arnold", r"\bkan\b", r"2.stage backprop",
        r"walkthrough.*neural", r"weight.*bias.*deep", r"weights.*activation",
        r"weight and bias", r"optimiz.*deep learn", r"loss function",
        r"cross.entropy", r"softmax", r"relu", r"sigmoid.*neural",
        r"training.*neural", r"training.*deep", r"optimize.*neural",
        r"ways to optimize neural", r"15 ways.*neural",
        r"artificial neural", r"artificial neuron",
        r"simple neural", r"a simple neural",
        r"batch processing.*neural", r"saturation.*neural",
        r"addressing saturation",
    ],
    "Machine-Learning": [
        r"machine learning", r"random forest", r"decision tree", r"support vector",
        r"\bsvm\b", r"\bknn\b", r"k.nearest", r"logistic regression", r"linear regression",
        r"naive bayes", r"xgboost", r"gradient boost", r"adaboost", r"catboost",
        r"lightgbm", r"ensemble", r"bagging", r"boosting", r"clustering",
        r"k.means", r"dbscan", r"hierarchical.*cluster", r"anomaly detection",
        r"outlier detect", r"feature.*select", r"feature.*engineer", r"feature.*import",
        r"feature.*scale", r"feature.*extract", r"cross.validation", r"overfitting",
        r"underfitting", r"bias.variance", r"regularization", r"lasso", r"ridge",
        r"elastic net", r"pca\b", r"principal component", r"dimensionality reduction",
        r"t.sne", r"umap\b", r"model.*select", r"model.*evaluat", r"model.*perform",
        r"model.*complex", r"hyperparameter", r"grid search", r"random search",
        r"bayesian optim", r"active learning", r"semi.supervised", r"self.supervised",
        r"data.*augment", r"class imbalance", r"imbalanced.*data", r"smote",
        r"roc.*curve", r"auc\b", r"precision.*recall", r"f1.*score",
        r"confusion matrix", r"accuracy.*metric", r"regression.*metric",
        r"mean.*error", r"mse\b", r"mae\b", r"rmse\b", r"mape\b",
        r"train.*test.*split", r"validation set", r"learning curve",
        r"ml.*interview", r"interview.*ml", r"interview.*data scien",
        r"acing.*interview", r"ml.*algorithm", r"most.*used.*algorithm",
        r"apriori algorithm", r"association rule", r"time series.*ml",
        r"xgboost.*time series", r"4 ways to test ml", r"ways to test ml",
        r"transfer learn(?!.*cnn)", r"advantages.*transfer learn",
        r"ml.*production", r"model.*production", r"4 ways to test",
        r"ml.*model", r"model.*ml", r"beginner.*machine learn",
        r"visual.*guide.*boost", r"visual.*boost", r"a visual.*machine",
    ],
    "Statistics-and-Math": [
        r"statistic", r"probabilit", r"bayesian", r"frequentist", r"hypothesis test",
        r"distribution", r"normal distribution", r"gaussian", r"poisson", r"binomial",
        r"t.test\b", r"z.test\b", r"chi.square", r"anova\b", r"p.value",
        r"confidence interval", r"linear algebra", r"matrix", r"eigenvalue",
        r"eigenvector", r"singular value", r"calculus", r"derivative",
        r"integral", r"differential equation", r"fourier", r"laplace",
        r"gradient.*math", r"hessian", r"optimization.*math", r"convex",
        r"information theory", r"entropy.*math", r"mutual information",
        r"causal inference", r"causal.*model", r"structural equation",
        r"time series.*statistic", r"statsmodel", r"bayes.*theorem",
        r"mathematical concept", r"mathematical definition", r"25 essential math",
        r"25 key math", r"essential.*math", r"algebraic", r"topolog",
        r"riemannian", r"manifold", r"galois", r"borel", r"jordan canonical",
        r"projective curve", r"scheme theory", r"syzygy", r"weakly differentiable",
        r"multiplicative number", r"complex analysis", r"real analysis",
        r"analysis i\b", r"partial differential", r"ordinary least squares",
        r"unbiased estimator", r"variance inflation", r"regression diagnostic",
        r"akaike", r"aic\b", r"bic\b", r"information criterion",
        r"skewness", r"kurtosis", r"empirical rule", r"central limit",
        r"law of large", r"monte carlo", r"viterbi", r"hidden markov",
        r"number theory", r"graph.*theor", r"network.*theor",
        r"why calculus", r"25.*math", r"normality test",
        r"data normality", r"r.squared", r"regression variance",
        r"accuracy vs.*precision", r"aggregation of reasoning",
        r"geometric deep learn", r"applying geometric",
    ],
    "Data-Engineering": [
        r"\bsql\b", r"database", r"postgresql", r"mysql", r"mongodb", r"redis",
        r"apache spark", r"pyspark", r"hadoop", r"kafka", r"airflow",
        r"etl\b", r"data pipeline", r"data warehouse", r"data lake",
        r"dbt\b", r"spark.*python", r"window function.*spark",
        r"user defined function.*spark", r"common table expression",
        r"cte\b", r"subquery", r"advanced sql", r"sql.*concept",
        r"sql.*technique", r"pandas.*polars.*sql", r"vanna.*sql",
        r"text.to.sql", r"scale.*database", r"database.*scale",
        r"7 strategies.*database",
    ],
    "Data-Visualization": [
        r"visualiz", r"matplotlib", r"seaborn", r"plotly", r"bokeh",
        r"altair", r"visual.*data", r"data.*visual", r"chart", r"plot\b",
        r"heatmap", r"histogram", r"scatter plot", r"line chart", r"bar.*plot",
        r"hexbin", r"treemap", r"dashboard", r"tableau", r"power bi",
        r"misleading.*visual", r"visual encoding", r"visual guide",
        r"alternatives.*bar", r"beyond bar", r"clutter.*plot",
        r"visual demonstration", r"animation.*cluster",
    ],
    "MLOps-and-Deployment": [
        r"mlops", r"docker", r"kubernetes", r"\baws\b", r"\bgcp\b", r"\bazure\b",
        r"cloud.*deploy", r"deploy.*cloud", r"model.*serv", r"model.*deploy",
        r"ci.cd", r"cicd", r"continuous integrat", r"continuous deploy",
        r"devops", r"inference.*produc", r"produc.*inference",
        r"batch.*inference", r"real.time.*inference", r"model.*monitor",
        r"model.*version", r"versioning.*ml", r"a.b.*test.*ml",
        r"feature store", r"experiment.*track", r"mlflow", r"weights.*biases.*tool",
        r"automating.*deploy", r"software deploy", r"crisis response.*ml",
        r"basic crisis", r"4 ways to test.*produc",
        r"vpc.*gateway", r"aws.*cost", r"cost.*aws",
    ],
    "Web-and-APIs": [
        r"fastapi", r"flask", r"django", r"\brest\b", r"api.*style",
        r"api.*architect", r"api.*perform", r"graphql", r"grpc",
        r"webhook", r"websocket", r"wsgi", r"asgi",
        r"web.*authen", r"session.*jwt", r"oauth", r"sso\b",
        r"microservice", r"monolith", r"http.*protocol",
        r"polling.*real.time", r"real.time.*communic",
        r"caching strateg", r"api.*caching", r"boost.*api",
        r"9 strategies.*api", r"9 strategies.*boost",
        r"architectural.*pattern.*python", r"architectural design pattern",
    ],
    "Data-Science": [
        r"data science", r"data analysis", r"exploratory", r"eda\b",
        r"pandas", r"numpy", r"data clean", r"missing data", r"missing value",
        r"handling missing", r"data.*preprocess", r"data.*transform",
        r"data.*manipulat", r"data.*wrangl", r"data.*quality",
        r"data.*import", r"data.*export", r"csv\b", r"json.*data",
        r"data.*type", r"data.*variable", r"11 types.*variable",
        r"data leakage", r"data split", r"train.*split",
        r"feature normali", r"feature standard", r"normalization",
        r"standardization", r"one.hot.*encod", r"label.*encod",
        r"target.*encod", r"polars", r"dask", r"modin",
        r"accelerate.*pandas", r"pandas.*parallel", r"pandas.*perform",
        r"pandas.*vectoriz", r"pandas.*optim", r"pandas.*trick",
        r"pandas.*techniqu", r"pandas.*manipulat", r"pandas.*concat",
        r"many.to.one.*pandas", r"avoiding.*pandas", r"advanced pandas",
        r"accessing data.*dict", r"user.*engagement.*churn",
        r"automated.*error.*tabular", r"automatic.*error.*tabular",
        r"financial data.*analyz", r"automated.*financial",
        r"churn.*python", r"analyzing.*lifetime",
        r"\$21 million", r"airline ticket",
        r"underrated.*package", r"10 underrated",
        r"7 essential.*finance", r"python.*finance",
        r"accelerating data process",
    ],
    "Python": [
        r"python", r"oop\b", r"object.oriented", r"magic method",
        r"dunder", r"decorator", r"generator", r"iterator", r"comprehension",
        r"dataclass", r"type hint", r"type annot", r"pydantic",
        r"lambda function", r"f.string", r"string.*format",
        r"logging.*python", r"debugging.*python", r"unittest", r"pytest",
        r"virtual env", r"packaging.*python", r"concurren", r"asyncio",
        r"threading", r"multiprocess", r"asyncio.*python",
        r"contextmanager", r"context manager", r"with statement",
        r"exception.*python", r"error.*handl.*python",
        r"memory.*python", r"garbage collect", r"weak reference",
        r"numba", r"cython", r"pypy",
        r"args.*kwargs", r"tuple.*unpack", r"asterisk.*python",
        r"underscore.*python", r"bitwise.*python",
        r"isinstance\(\)", r"isinstance.*python", r"iter\(\).*sentinel",
        r"override.*python", r"self type.*python", r"@override",
        r"mutable default", r"persistent.*object", r"reference cycle",
        r"python.*interview", r"interview.*python",
        r"one.liner.*python", r"python.*one.liner",
        r"helpful.*python", r"python.*concept",
        r"advantages.*f.string", r"advantages.*pure python",
        r"command.*line.*python", r"python.*flag",
        r"functional.*program.*python", r"function.*python",
        r"advanced.*python", r"python.*advanced",
        r"beginner.*python", r"python.*beginner",
        r"3 approaches.*concurren",
    ],
}

FOLDER_ORDER = [
    "LLMs-and-GenAI",
    "NLP-and-Transformers",
    "Computer-Vision",
    "Reinforcement-Learning",
    "Deep-Learning",
    "Data-Visualization",
    "Data-Engineering",
    "MLOps-and-Deployment",
    "Web-and-APIs",
    "Machine-Learning",
    "Statistics-and-Math",
    "Data-Science",
    "Python",
]


def classify(filename):
    name = filename.replace(".md", "").lower()
    for folder in FOLDER_ORDER:
        patterns = CATEGORIES[folder]
        for pat in patterns:
            if re.search(pat, name, re.IGNORECASE):
                return folder
    return "Misc"


def main():
    files = [f for f in os.listdir(BASE) if f.endswith(".md") and os.path.isfile(os.path.join(BASE, f))]

    # Count per category first
    counts = {folder: [] for folder in FOLDER_ORDER}
    counts["Misc"] = []
    for f in files:
        cat = classify(f)
        counts[cat].append(f)

    print(f"\nTotal files: {len(files)}\n")
    print("Distribution:")
    for cat in FOLDER_ORDER + ["Misc"]:
        print(f"  {cat:<30} {len(counts[cat]):>5} files")

    print("\nMisc files:")
    for f in sorted(counts["Misc"]):
        print(f"  {f}")

    # Create folders and move files
    all_folders = FOLDER_ORDER + ["Misc"]
    for folder in all_folders:
        folder_path = os.path.join(BASE, folder)
        os.makedirs(folder_path, exist_ok=True)

    moved = 0
    for folder in all_folders:
        for f in counts[folder]:
            src = os.path.join(BASE, f)
            dst = os.path.join(BASE, folder, f)
            shutil.move(src, dst)
            moved += 1

    print(f"\nMoved {moved} files into folders.")


if __name__ == "__main__":
    main()
