"""
Extended benchmark tasks — harder multi-step, real-world scenarios.

Categories added:
  - rag_retrieval: Tasks requiring document ingestion + retrieval
  - multi_source_research: Tasks requiring 5+ tool calls across sources
  - error_recovery_stress: Tasks designed to trigger specific failure modes
  - data_pipeline: Tasks chaining search → extract → compute → write
"""

from __future__ import annotations

from evaluation.benchmarks.tasks import (
    BenchmarkTask,
    Difficulty,
    ExpectedStep,
    TaskCategory,
)


# ═══════════════════════════════════════════════════════════════════════
# New category for RAG tasks
# ═══════════════════════════════════════════════════════════════════════

class ExtendedCategory:
    RAG_RETRIEVAL = "rag_retrieval"
    MULTI_SOURCE_RESEARCH = "multi_source_research"
    ERROR_RECOVERY_STRESS = "error_recovery_stress"
    DATA_PIPELINE = "data_pipeline"


EXTENDED_TASKS: list[BenchmarkTask] = [
    # ── RAG Retrieval Tasks ──────────────────────────────────────
    BenchmarkTask(
        task_id="rag_001",
        description=(
            "Search for a recent article about climate change impacts on agriculture, "
            "fetch the full article, ingest it into the knowledge base, then use "
            "semantic search to answer: 'What crops are most affected by rising temperatures?'"
        ),
        category=TaskCategory.WEB_NAVIGATION,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(
                tool="web_search",
                description="Search for climate agriculture article",
                required_args=["query"],
            ),
            ExpectedStep(
                tool="web_fetch",
                description="Fetch the full article",
                required_args=["url"],
            ),
            ExpectedStep(
                tool="rag_ingest",
                description="Ingest article into knowledge base",
                required_args=["text"],
            ),
            ExpectedStep(
                tool="rag_search",
                description="Search for crop impact information",
                required_args=["query"],
            ),
        ],
        success_criteria="Retrieves relevant crop data from ingested article",
        expected_final_answer_contains=["crop", "temperature"],
        tags=["rag", "multi_step"],
    ),

    BenchmarkTask(
        task_id="rag_002",
        description=(
            "Ingest the following document into the knowledge base, then answer "
            "questions about it using RAG retrieval:\n\n"
            "Document: 'The Python programming language was created by Guido van Rossum "
            "and first released in 1991. Python 3.12 introduced several performance "
            "improvements including a specializing adaptive interpreter. The language "
            "supports multiple paradigms including procedural, object-oriented, and "
            "functional programming. Python's package manager pip has over 400,000 "
            "packages available on PyPI.'\n\n"
            "Question: When was Python first released and who created it?"
        ),
        category=TaskCategory.MULTI_STEP_REASONING,
        difficulty=Difficulty.MEDIUM,
        expected_steps=[
            ExpectedStep(
                tool="rag_ingest",
                description="Ingest the Python document",
                required_args=["text"],
            ),
            ExpectedStep(
                tool="rag_search",
                description="Search for creation/release info",
                required_args=["query"],
            ),
        ],
        success_criteria="Answers: Guido van Rossum, 1991",
        expected_final_answer_contains=["Guido", "1991"],
        tags=["rag", "factual"],
    ),

    BenchmarkTask(
        task_id="rag_003",
        description=(
            "Search for information about three different machine learning frameworks "
            "(PyTorch, TensorFlow, JAX), fetch details about each, ingest all three "
            "into the knowledge base, then use RAG to generate a comparison summary."
        ),
        category=TaskCategory.INFORMATION_SYNTHESIS,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(tool="web_search", description="Search PyTorch", required_args=["query"]),
            ExpectedStep(tool="web_fetch", description="Fetch PyTorch details", required_args=["url"], order_flexible=True),
            ExpectedStep(tool="web_search", description="Search TensorFlow", required_args=["query"]),
            ExpectedStep(tool="web_fetch", description="Fetch TensorFlow details", required_args=["url"], order_flexible=True),
            ExpectedStep(tool="web_search", description="Search JAX", required_args=["query"]),
            ExpectedStep(tool="rag_ingest", description="Ingest framework docs", required_args=["text"], order_flexible=True),
            ExpectedStep(tool="rag_context", description="Retrieve comparison context", required_args=["query"]),
        ],
        success_criteria="Compares all three frameworks with specific details",
        expected_final_answer_contains=["PyTorch", "TensorFlow"],
        max_acceptable_steps=12,
        tags=["rag", "multi_source", "synthesis"],
    ),

    # ── Multi-Source Research Tasks ──────────────────────────────
    BenchmarkTask(
        task_id="research_001",
        description=(
            "Research the top 3 electric vehicle manufacturers by global sales in 2024. "
            "For each, find their flagship model, starting price, and range. "
            "Calculate the average price across all three, then write a structured "
            "comparison report to a file."
        ),
        category=TaskCategory.INFORMATION_SYNTHESIS,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(tool="web_search", description="Search EV manufacturers", required_args=["query"]),
            ExpectedStep(tool="web_search", description="Search manufacturer 1 details", required_args=["query"]),
            ExpectedStep(tool="web_search", description="Search manufacturer 2 details", required_args=["query"]),
            ExpectedStep(tool="web_search", description="Search manufacturer 3 details", required_args=["query"]),
            ExpectedStep(tool="calculator", description="Calculate average price", required_args=["expression"]),
            ExpectedStep(tool="file_write", description="Write comparison report", required_args=["filename", "content"]),
        ],
        success_criteria="Identifies top 3 EV makers with correct details and average",
        expected_final_answer_contains=["Tesla", "price", "range"],
        max_acceptable_steps=10,
        tags=["research", "multi_source"],
    ),

    BenchmarkTask(
        task_id="research_002",
        description=(
            "Find the current exchange rates for USD to EUR, GBP, and JPY. "
            "Calculate how much 10,000 USD would be in each currency. "
            "Then determine which currency has appreciated the most against "
            "USD over the past year by searching for historical rates."
        ),
        category=TaskCategory.INFORMATION_SYNTHESIS,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(tool="web_search", description="Current USD/EUR rate", required_args=["query"]),
            ExpectedStep(tool="web_search", description="Current USD/GBP rate", required_args=["query"]),
            ExpectedStep(tool="web_search", description="Current USD/JPY rate", required_args=["query"]),
            ExpectedStep(tool="calculator", description="Convert to EUR", required_args=["expression"]),
            ExpectedStep(tool="calculator", description="Convert to GBP", required_args=["expression"]),
            ExpectedStep(tool="calculator", description="Convert to JPY", required_args=["expression"]),
            ExpectedStep(tool="web_search", description="Historical exchange rates", required_args=["query"]),
        ],
        success_criteria="Correct conversions and identifies appreciation trend",
        expected_final_answer_contains=["EUR", "GBP", "JPY", "10,000"],
        max_acceptable_steps=12,
        tags=["research", "calculation", "multi_step"],
    ),

    BenchmarkTask(
        task_id="research_003",
        description=(
            "Research the population, area, and GDP of India, Brazil, and Nigeria. "
            "Calculate population density and GDP per capita for each. "
            "Rank them by GDP per capita and write the analysis to a file."
        ),
        category=TaskCategory.INFORMATION_SYNTHESIS,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(tool="web_search", description="India stats", required_args=["query"]),
            ExpectedStep(tool="web_search", description="Brazil stats", required_args=["query"]),
            ExpectedStep(tool="web_search", description="Nigeria stats", required_args=["query"]),
            ExpectedStep(tool="calculator", description="Calculate densities", required_args=["expression"]),
            ExpectedStep(tool="calculator", description="Calculate GDP per capita", required_args=["expression"]),
            ExpectedStep(tool="file_write", description="Write analysis", required_args=["filename", "content"]),
        ],
        success_criteria="Correct calculations and ranking by GDP per capita",
        expected_final_answer_contains=["density", "per capita"],
        max_acceptable_steps=12,
        tags=["research", "calculation"],
    ),

    # ── Data Pipeline Tasks ──────────────────────────────────────
    BenchmarkTask(
        task_id="pipeline_001",
        description=(
            "Search for the latest unemployment rates for the G7 countries "
            "(US, UK, Canada, France, Germany, Italy, Japan). "
            "Extract the numbers, calculate the G7 average, identify which "
            "country has the lowest and highest rate, and save the analysis."
        ),
        category=TaskCategory.CALCULATION,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(tool="web_search", description="G7 unemployment rates", required_args=["query"]),
            ExpectedStep(tool="web_fetch", description="Fetch detailed data", required_args=["url"], order_flexible=True),
            ExpectedStep(tool="text_analysis", description="Extract key facts", required_args=["text", "operation"]),
            ExpectedStep(tool="calculator", description="Calculate average", required_args=["expression"]),
            ExpectedStep(tool="file_write", description="Save analysis", required_args=["filename", "content"]),
        ],
        success_criteria="Correct average with highest/lowest identification",
        expected_final_answer_contains=["average", "unemployment"],
        max_acceptable_steps=10,
        tags=["pipeline", "data_extraction"],
    ),

    BenchmarkTask(
        task_id="pipeline_002",
        description=(
            "Find the top 5 most-watched YouTube videos of all time. "
            "For each, find the view count, then calculate the total "
            "combined views and the percentage each video represents "
            "of the total. Write the results to a file."
        ),
        category=TaskCategory.INFORMATION_SYNTHESIS,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(tool="web_search", description="Top YouTube videos", required_args=["query"]),
            ExpectedStep(tool="web_fetch", description="Get detailed view counts", required_args=["url"], order_flexible=True),
            ExpectedStep(tool="calculator", description="Calculate total views", required_args=["expression"]),
            ExpectedStep(tool="calculator", description="Calculate percentages", required_args=["expression"]),
            ExpectedStep(tool="file_write", description="Write results", required_args=["filename", "content"]),
        ],
        success_criteria="Lists top 5 videos with correct percentage calculations",
        expected_final_answer_contains=["views", "percent"],
        max_acceptable_steps=10,
        tags=["pipeline", "data_extraction"],
    ),

    # ── Error Recovery Stress Tests ──────────────────────────────
    BenchmarkTask(
        task_id="stress_001",
        description=(
            "Search for 'quantum computing recent breakthroughs', fetch the "
            "top result, analyze it, then search for 'classical computing comparison', "
            "fetch that result, and synthesize both into a comparison. "
            "If any fetch fails, recover by using search snippets instead."
        ),
        category=TaskCategory.MULTI_STEP_REASONING,
        difficulty=Difficulty.HARD,
        expected_steps=[
            ExpectedStep(tool="web_search", description="Quantum computing search", required_args=["query"]),
            ExpectedStep(tool="web_fetch", description="Fetch quantum article", required_args=["url"]),
            ExpectedStep(tool="text_analysis", description="Analyze quantum content", required_args=["text", "operation"]),
            ExpectedStep(tool="web_search", description="Classical computing search", required_args=["query"]),
            ExpectedStep(tool="web_fetch", description="Fetch classical article", required_args=["url"]),
            ExpectedStep(tool="text_analysis", description="Analyze classical content", required_args=["text", "operation"]),
        ],
        success_criteria="Produces comparison even if some fetches fail",
        expected_final_answer_contains=["quantum", "classical"],
        max_acceptable_steps=10,
        tags=["error_recovery", "resilience"],
    ),

    BenchmarkTask(
        task_id="stress_002",
        description=(
            "Calculate the compound interest on $50,000 at 7.5% annual rate "
            "compounded monthly for 10 years. Then calculate for quarterly "
            "and annual compounding. Compare all three and determine the "
            "difference between monthly and annual compounding."
        ),
        category=TaskCategory.CALCULATION,
        difficulty=Difficulty.MEDIUM,
        expected_steps=[
            ExpectedStep(tool="calculator", description="Monthly compounding", required_args=["expression"]),
            ExpectedStep(tool="calculator", description="Quarterly compounding", required_args=["expression"]),
            ExpectedStep(tool="calculator", description="Annual compounding", required_args=["expression"]),
            ExpectedStep(tool="calculator", description="Difference calculation", required_args=["expression"]),
        ],
        success_criteria="Correct compound interest for all three and difference",
        expected_final_answer_contains=["compound", "monthly", "annual"],
        tags=["calculation", "multi_step"],
    ),

    BenchmarkTask(
        task_id="stress_003",
        description=(
            "Perform sentiment analysis on these three reviews, then calculate "
            "the overall sentiment distribution:\n"
            "1. 'This product is absolutely amazing and the best I have ever used!'\n"
            "2. 'Terrible experience, the worst customer service and poor quality.'\n"
            "3. 'It works fine, nothing special but gets the job done.'\n"
            "Write a summary report with sentiment percentages."
        ),
        category=TaskCategory.MULTI_STEP_REASONING,
        difficulty=Difficulty.MEDIUM,
        expected_steps=[
            ExpectedStep(tool="text_analysis", description="Analyze review 1", required_args=["text", "operation"],
                         expected_arg_patterns={"operation": "sentiment"}),
            ExpectedStep(tool="text_analysis", description="Analyze review 2", required_args=["text", "operation"]),
            ExpectedStep(tool="text_analysis", description="Analyze review 3", required_args=["text", "operation"]),
            ExpectedStep(tool="file_write", description="Write summary", required_args=["filename", "content"]),
        ],
        success_criteria="Correct sentiment for each review and distribution percentages",
        expected_final_answer_contains=["positive", "negative", "neutral"],
        tags=["text_analysis", "multi_step"],
    ),
]


def get_all_tasks_with_extended() -> list[BenchmarkTask]:
    """Return original + extended benchmark tasks."""
    from evaluation.benchmarks.tasks import BENCHMARK_TASKS
    return BENCHMARK_TASKS + EXTENDED_TASKS
