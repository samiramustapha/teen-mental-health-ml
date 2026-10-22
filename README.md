# Teen Mental Health & Social Media

Predicts teen depression risk from social media use, sleep, academic performance,
and stress indicators, then segments the results by platform, age, sleep, and
physical activity. Includes an interactive CLI for exploring subgroups.

## The result that matters

The classifier scores **99% test accuracy**. That number is close to meaningless.

**97% of the dataset is the majority class**, so a model that predicts "not
depressed" for every single teen also scores 97%. Measured against that
baseline, 99% accuracy is a 2-point improvement, not a working model.

The metric that actually describes the model is **recall on the at-risk class:
64%**. The model exists to find depressed teens, and it misses roughly a third
of them. For a screening tool, a false negative is the expensive error — it
costs someone an intervention — so recall on the minority class is the binding
constraint, and accuracy is a distraction.

The confusion matrix is in the output figure for exactly this reason.

## Method

- **Features (8):** daily social media hours, sleep hours, screen time before
  sleep, academic performance, physical activity, stress level, anxiety level,
  addiction level
- **Target:** `depression_label` (binary)
- **Model:** `MLPClassifier`, two hidden layers of 8 units, max 1000 iterations
- **Split:** 75/25 train/test, fixed random seed for reproducibility
- **Scaling:** `StandardScaler` **fit on the training set only**, then applied to
  both splits. Fitting the scaler on the full dataset before splitting leaks test
  distribution into training and inflates the score — a common and easy mistake.

## Descriptive analysis

Before modeling, the script works through the data directly: stress by addiction
level, anxiety by daily usage, depression rate by usage band, usage by gender,
depression rate per platform, the effect of sleep and physical activity on
anxiety and depression, and GPA split at the median of daily usage.

## Running it

```bash
pip install -r requirements.txt
python teen_mental_health.py
```

**The dataset is not in this repo.** `Teen_Mental_Health_Dataset.csv` is a
third-party dataset and is not redistributed here. Download it separately and
place it in the project root before running.

Expected columns: `age`, `gender`, `platform_usage`,
`daily_social_media_hours`, `sleep_hours`, `screen_time_before_sleep`,
`academic_performance`, `physical_activity`, `stress_level`, `anxiety_level`,
`addiction_level`, `depression_label`.

## Known limitations

- **Class imbalance is unaddressed.** No class weighting, resampling, or decision
  threshold tuning. These are the obvious next steps for lifting recall on the
  at-risk class, and the reason the current recall is 64%.
- **The findings are correlational.** Nothing here supports a causal claim about
  social media and depression.
- **Single train/test split.** No cross-validation, so the reported figures carry
  more variance than they appear to.

---

Built for TAC 216 (Applied Python / Machine Learning), University of Southern
California.
