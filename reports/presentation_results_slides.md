# Presentation: Results Section (Slides & Speaker Notes)

Text for the Results section of the [Figma deck](https://www.figma.com/slides/rplR6WAbzbSNcrZ9wMQPnW), ready to paste.
Numbers come from the ranking models in #3 (best model: Random Forest, Full config), run on the 10% user sample (1,968 test users). Charts: `reports/figures/`.

Design: duplicate an existing slide so the style carries over (noted per slide). Cards = title, big number, heading, body.

## Section order

| # | Slide | Status |
|---|---|---|
| 10 | How We Train and Test | new |
| 11 | Results | in Figma |
| 12 | Analysis | in Figma |
| 13 | Where the Model Struggles | new |
| 14 | Limitations | new |
| 15 | Conclusion & Next Steps | in Figma |
| 16 | References | new |
| 17 | Thank You | new |

---

## 10. How We Train and Test

*Duplicate the Results slide (3 cards).*

**Title:** How We Train and Test
**Subtitle:** Hide one city per user, then ask the model to find it.

| Card | Number | Heading | Body |
|---|---|---|---|
| 1 (highlighted) | 1 city | hidden per user | One city the user really visited. Everything else is their history. |
| 2 | 4 | unvisited cities | Sampled as negatives. The model learns visited (1) vs not (0). |
| 3 | 70/10/20 | train / val / test | Users split three ways. No user appears in two sets. |

**Takeaway:** Test: rank ~276 cities per user and check if the hidden city is in the top 10.

**Speaker notes**
- This is leave-one-city-out evaluation: the model never sees the city it must find.
- User features come only from history, so the answer can't leak in.
- Validation users (10%) are used to tune settings; test users (20%) are used once, at the end.
- Same test users for every model, so results are directly comparable.

---

## 11. Results

**Title:** Results
**Subtitle:** Recall@10 on 1,968 test users: is the visited city in the top 10 of ~276?

| Card | Number | Heading | Body |
|---|---|---|---|
| 1 (highlighted) | 0.725 | Random Forest | Best model. The visited city is in the top 10 for 72% of users. |
| 2 | 0.634 | Two-Tower | Her two-tower network, retrained to rank cities. Beats the baseline too. |
| 3 | 0.358 | Popularity baseline | Ranks cities by size and rating. Random guessing scores 0.036. |

**Takeaway:** The Random Forest doubles the baseline: 0.725 vs 0.358.

**Chart (optional):** `reports/figures/recall_by_model.png`

**Speaker notes**
- Same 1,968 test users for every model, so the comparison is fair.
- Recall@10: for each user we hide one city they visited, rank ~276 cities, and check if it lands in the top 10.
- **Random Forest 0.725** · NDCG@10 0.543 · MRR@10 0.486. Two-Tower 0.634 · NDCG 0.432.
- Sanity checks: random scores 0.038 (expected ~0.036) and Popularity reproduces 0.358.
- Settings (tree depth, best epoch) were tuned on 984 separate validation users, not on the test users.
- Transition: so why does the model do so well? Next slide.

---

## 12. Analysis

**Title:** Analysis
**Subtitle:** What drives the ranking, and does weather help?

| Card | Number | Heading | Body |
|---|---|---|---|
| 1 (highlighted) | 43% | distance from home | Top feature in the Random Forest, followed by city size. |
| 2 | 81% | of trips are local | Hidden cities are within 50 km of home (median 19 km). |
| 3 | +0.002 | from review text | Weather (0.723 → 0.720) and review embeddings (→ 0.725) barely move it. |

**Takeaway:** The model learns "big cities near home". Weather can't help local trips.

**Speaker notes**
- Feature importance: distance 43%, city review volume 17%, city size 15%, temperature vs home 7%.
- 81% of hidden cities are within 50 km of the user's home city, so most "cross-city" reviews are suburb hops (e.g. Reno → Sparks).
- That is why distance dominates and why nearby big cities are easy to predict.
- **RQ1 (weather):** no gain, 0.723 → 0.720. Local trips share the home climate, so weather has nothing to add.
- **RQ2/RQ3 (review text, combined):** 0.725 for both; text adds little once distance and city size are known.
- Weather now uses Yelp business coordinates; name-based geocoding had placed some cities on the wrong continent.
- **RQ4 (models):** the Random Forest beats the Two-Tower with these 13 simple features.

---

## 13. Where the Model Struggles

*Duplicate the Analysis slide.*

**Title:** Where the Model Struggles
**Subtitle:** Random Forest Recall@10 by distance from home to the hidden city.

| Card | Number | Heading | Body |
|---|---|---|---|
| 1 (highlighted) | 0.766 | under 50 km | 1,597 users. Nearby big cities are easy to predict. |
| 2 | 0.588 | over 500 km | 320 users. Long trips still work, often to well-known cities. |
| 3 | 0.275 | 50–500 km | Only 51 users. Regional trips are the hardest case. |

**Takeaway:** Strong on local trips, weaker on regional ones: real travel needs more than distance.

**Speaker notes**
- Overall Recall@10 is 0.725, but it depends on how far the trip is.
- Local trips (81% of users) drive the high score: distance alone points to the right suburb.
- Long trips (over 500 km) still reach 0.588, likely because travelers pick famous cities.
- Regional trips (50–500 km) are weakest at 0.275, but only 51 users, so treat it as a hint, not a result.
- This is where weather and review text should help most: next experiment.

---

## 14. Limitations

*Duplicate the Conclusion slide (numbered layout).*

**Title:** Limitations

| # | Heading | Body |
|---|---|---|
| 1 | Limited geography | Yelp covers about 11 US and Canadian metro areas, not global travel. |
| 2 | A review is not a trip | Users may review places near home without traveling. |
| 3 | Suburbs count as cities | 81% of hidden cities are within 50 km of home. |
| 4 | Duplicate city spellings | "St. Louis" / "Saint Louis" count as different cities; affects 7.1% of test users and slightly inflates Recall@10. |

**Speaker notes**
- Our "destinations" are Yelp cities, so the results describe US/Canada metro areas only.
- We infer trips from reviews; there is no ground truth of actual travel.
- Because suburbs are separate cities, many "trips" are local hops. Grouping into metro areas is the planned fix.
- Numbers come from a 10% user sample (1,968 test users); the full-data run is next.

---

## 15. Conclusion & Next Steps

**Title:** Conclusion & Next Steps

| # | Heading | Body |
|---|---|---|
| 1 | Ranking models beat the baseline | Random Forest Recall@10 = 0.725 and Two-Tower = 0.634, vs Popularity 0.358. |
| 2 | Weather doesn't help yet | Most trips are local, so they share the home city's climate (0.723 → 0.720). |
| 3 | Next: real travel and review text | Group suburbs into metro areas, add review embeddings, run on all users. |

**Speaker notes**
- Answer to the main question so far: a personalized ranking model clearly beats popularity (about 2×).
- Weather (RQ1) shows no gain on this data; we expect it to matter only for real long-distance travel.
- Review embeddings (RQ2) and the combined model (RQ3) are the next experiments.
- Grouping suburbs into metro areas will test whether the model still works when distance can't do the work.
- Close: thank the audience and open for questions.

---

## 16. References

*Duplicate the Evaluation Metrics slide.*

1. Borràs, J., Moreno, A., & Valls, A. (2014). Intelligent tourism recommender systems: A survey. Expert Systems with Applications.
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP.
3. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural collaborative filtering. WWW.
4. Yelp Open Dataset. yelp.com/dataset
5. Open-Meteo Historical Weather API. open-meteo.com

---

## 17. Thank You

*Duplicate the cover slide.*

**Title:** Thank You
**Subtitle:** Questions?
**Footer:** Group 8 · Nomin Nergui · Tumenjargal Altanginj · Temuujin Bat Amgalan · Temuulen Khuchit
