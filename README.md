# ADS-B Trajectory Analytics — Streamlit

## გაშვება
```bash
pip install -r requirements.txt
streamlit run app.py
```
საიდბარში ატვირთეთ `adsb_hexagon_data.csv` (ან მიუთითეთ ლოკალური გზა). ატვირთვის გარეშე ჩანს დემო (`data.json`).
თუ CSV ძალიან დიდია (iq/magnitudes/phases სვეტების გამო), ლოკალური გზა უფრო სწრაფია, ვიდრე ატვირთვა.

## ონლაინ გამოქვეყნება (უფასო)
1. ატვირთეთ ფაილები GitHub-ზე (`app.py`, `prepare_data.py`, `requirements.txt`, `data.json`).
2. share.streamlit.io → New app → აირჩიეთ repo და `app.py`.
დიდი CSV GitHub-ზე არ ჩადოთ: ლოკალურად გაუშვით `python prepare_data.py adsb_hexagon_data.csv`, რომ მივიღოთ მსუბუქი `data.json`, და ის ატვირთეთ.

## ფაილები
- `app.py` — Streamlit ინტერფეისი (2D, PyDeck, 3D, პროფილები, ანომალიები)
- `prepare_data.py` — დამუშავების ლოგიკა (ფრენები, სიჩქარე, track, Isolation Forest)
- `data.json` — დემო მონაცემები
