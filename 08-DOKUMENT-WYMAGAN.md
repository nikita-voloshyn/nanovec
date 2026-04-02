# Dokument wymagań projektowych

**Nazwa projektu:** NanoVec — lekka, pamięciowa baza danych wektorowych dla agentów AI

---

## 0. Wersje dokumentu

| Wersja | Data | Zakres zmian |
|--------|------|-------------|
| 1.0 | 2026-03-18 | Wersja inicjalna |

---

## 1. Elementy składowe projektu


---

## 2. Granice projektu

### Co jest w zakresie

- Dokładne wyszukiwanie KNN (100% recall) metodą brute-force SIMD i KD-Tree.
- Integracja MCP w trybie stdio (lokalne środowisko) i SSE (środowisko rozproszone).
- Obsługa wymiarów 1–4096 (przy dostępnej pamięci RAM).
- Obsługa embeddingów z OpenAI API (`text-embedding-3-small`, `ada-002`).
- Działanie na Linux x86_64 i macOS ARM64 (Apple Silicon).

### Co jest poza zakresem (z uzasadnieniem)

| Funkcjonalność | Uzasadnienie wykluczenia |
|---------------|------------------------|
| **Trwałość danych (disk persistence)** | Sprzeczna z architektoniczną ideą projektu: NanoVec jest celowo efemeryczny. Dodanie persystencji wymagałoby mechanizmów WAL, replikacji i migracji schematów — zamieniając projekt w kolejną konkurencję dla Qdrant. |
| **Przybliżone wyszukiwanie HNSW** | Złożoność implementacji (~120 rh) wykracza poza zakres jednosemestralnego projektu. KD-Tree pokrywa use case dla niskich wymiarów; brute-force SIMD jest praktycznie optymalny dla 768-dim. |
| **Autentykacja w trybie stdio** | Stdio jest chronione przez mechanizmy IPC systemu operacyjnego. Autentykacja nie jest wymagana gdy klient i serwer działają w tym samym procesie OS. |
| **Obsługa lokalnych modeli embeddingów (candle)** | Wymagałoby dodania ciężkich zależności (candle-core, tokenizers), co sprzeciwia się filozofii zero-dependency. |
| **Obsługa Windows** | Zestaw narzędzi musl/static linking jest skomplikowany na Windows. Projekt celuje w środowiska deweloperskie (Linux/macOS). |
| **Filtrowanie metadanych podczas wyszukiwania** | Wyszukiwanie hybrydowe (wektor + filtr SQL-like) wymaga dodatkowego query planner. Zakres przekracza semestralne ramy. |
| **GUI / dashboard** | NanoVec jest biblioteką/serwerem — interfejs użytkownika nie jest częścią wartości projektu. |

---

## 3. Lista wymagań funkcjonalnych

### RF-01 — Wstawianie wektora
**User story**: Jako agent AI, chcę wstawić wektor f32 wraz z metadanymi tekstowymi, aby móc go później odnaleźć przez wyszukiwanie semantyczne.
- Wejście: `Vec<f32>` o stałej wymiarowości `D` + `String` (metadane)
- Wyjście: `usize` (indeks wektora w zbiorze)
- Walidacja: odrzucenie wektora o wymiarowości ≠ D

### RF-02 — Wyszukiwanie K najbliższych sąsiadów (KNN) — brute-force
**User story**: Jako agent AI, chcę wyszukać K wektorów najbliższych zapytaniu, uzyskując wyniki w czasie < 10 ms dla 10 000 wektorów.
- Wejście: `query: &[f32]`, `k: usize`
- Wyjście: `Vec<(f32, usize)>` — posortowane (odległość, indeks)
- Metryki: L2 squared, cosine distance

### RF-03 — Wyszukiwanie KNN — KD-Tree
**User story**: Jako agent AI z małym wymiarem wektorów (D ≤ 20), chcę wyszukiwania KNN z przycinaniem gałęzi dla przyspieszenia O(log n).
- Wejście: `query: &[f32]`, `k: usize`
- Wyjście: identyczne jak RF-02
- Warunek: wyniki muszą być identyczne jak brute-force (100% recall)

### RF-04 — Adaptacyjny wybór strategii wyszukiwania
**User story**: Jako deweloper, chcę aby system automatycznie wybierał optymalną strategię, bez konieczności ręcznej konfiguracji.
- Logika: `D > 20` lub `count < 10 000` → brute-force SIMD; `D ≤ 20` i `count ≥ 10 000` → KD-Tree

### RF-05 — Usuwanie wektora po ID
**User story**: Jako agent AI, chcę usunąć konkretny dokument z pamięci po jego ID.
- Wejście: `id: u64`
- Wyjście: `Result<(), DeleteError>`

### RF-06 — Budowa indeksu KD-Tree
**User story**: Jako system, chcę zbudować indeks przestrzenny po wsadowym wstawieniu wektorów, aby przyspieszyć kolejne zapytania.
- Złożoność: O(n log n)
- Algorytm wyboru mediany: quickselect O(n)

### RF-07 — Narzędzie MCP: `index_document`
**User story**: Jako agent AI, chcę zindeksować tekst dokumentu przez wywołanie narzędzia MCP.
- Chunking tekstu na fragmenty (domyślnie 500 znaków, granica zdania)
- Wsadowe generowanie embeddingów (OpenAI API)
- Wstawienie wszystkich wektorów do bazy
- Wyjście: `{document_id, chunks_indexed, total_vectors}`

### RF-08 — Narzędzie MCP: `semantic_search`
**User story**: Jako agent AI, chcę semantycznie przeszukać zindeksowane dokumenty przez narzędzie MCP.
- Embedding zapytania → KNN search → filtrowanie po `min_score`
- Wyjście: `{results: [{id, score, text, metadata}], query_time_ms}`

### RF-09 — Narzędzie MCP: `delete_document`
**User story**: Jako agent AI, chcę usunąć dokument po ID lub filtrze metadanych.
- Usunięcie po `document_id` lub `metadata_filter`
- Po usunięciu: przebudowa indeksu KD-Tree

### RF-10 — Transport stdio
**User story**: Jako programista używający Claude Desktop, chcę podłączyć NanoVec jako serwer MCP przez stdin/stdout.
- Wszystkie logi kierowane wyłącznie na stderr (nie wolno zanieczyszczać stdout)
- Konfiguracja przez `claude_desktop_config.json`

### RF-11 — Transport SSE
**User story**: Jako programista systemów rozproszonych, chcę udostępnić NanoVec przez HTTP Server-Sent Events.
- Endpoint: `GET /sse`
- Serwer: axum + tower-http

### RF-12 — Obsługa błędów
- `EmptyDatabase` — wyszukiwanie na pustym zbiorze
- `InvalidDimension {expected, got}` — niezgodna wymiarowość
- `IndexNotBuilt` — fallback na brute-force zamiast paniki
- `EmbeddingCallFailed` — błąd sieci / API

---

## 4. Lista wymagań niefunkcjonalnych

### RN-01 — Latencja wyszukiwania
Czas zapytania KNN (k=10) dla zbioru 10 000 wektorów 768-dim: **< 500 μs** (brute-force SIMD), mierzone na Apple M1 Pro lub Intel Core i7 10. generacji.

### RN-02 — Szybkość indeksowania
Wstawianie wektorów (bez embeddingu): **> 100 000 vec/s** przy wymiarowości 768.

### RN-03 — Precyzja wyników
Brute-force i KD-Tree muszą zwracać identyczne wyniki (recall = 100%), weryfikowane przez testy property-based (`proptest`).

### RN-04 — Zużycie pamięci
Dla 10 000 wektorów 768-dim: całkowite zużycie RAM **< 50 MB** (wektory ~30,7 MB + narzut).

### RN-05 — Kompilacja i zależności runtime
Projekt kompiluje się bez błędów na stable Rust 1.75+ (tryb nightly wymagany tylko dla `portable_simd`). Plik binarny release nie wymaga żadnych bibliotek dynamicznych (statyczne linkowanie).

### RN-06 — Bezpieczeństwo typów MCP
Wszystkie narzędzia MCP posiadają kompletne schematy JSON Schema. Nieprawidłowe wywołania są odrzucane przed dotarciem do silnika bazy danych.

### RN-07 — Prywatność danych
Zero bajtów danych wektorowych zapisanych na dysku podczas normalnej pracy. Zakończenie procesu gwarantuje usunięcie wszystkich danych z RAM.

### RN-08 — Przenośność
Kompilacja i testy przechodzą na: Ubuntu 22.04 x86_64, macOS 14 ARM64. CI: GitHub Actions (matrix: `[ubuntu-latest, macos-latest] × [stable, nightly]`).

### RN-09 — Obsługa brzegowych wartości zmiennoprzecinkowych
Cosine similarity zwraca 0.0 dla wektorów zerowych (zamiast NaN lub panic). Wyniki cosine clampowane do [-1.0, 1.0].

### RN-10 — Czas uruchomienia
Serwer MCP gotowy na przyjęcie pierwszego zapytania w ciągu **< 100 ms** od uruchomienia procesu.

---

## 5. Mierzalne wskaźniki wdrożeniowe

1. **Wdrożenie jako serwer MCP**: Binarny plik `nanovec` zostanie opublikowany jako GitHub Release dla Linux x86_64 i macOS ARM64. Konfiguracja Claude Desktop opisana w dokumentacji. System zostanie przetestowany z Claude Desktop — co najmniej **10 sesji** z indeksowaniem dokumentów i wyszukiwaniem semantycznym.

2. **Wydajność**: W raportach benchmarkowych (criterion) akceleracja SIMD wyniesie co najmniej **4× powyżej baseline skalarnego** dla 768-dim wektorów na każdej docelowej platformie.

3. **Poprawność**: Testy `proptest` (≥ 1 000 losowych przypadków) potwierdzą, że SIMD i KD-Tree zwracają wyniki zgodne ze skalarnym baseline z tolerancją `ε = 1e-4`.

4. **Testy end-to-end MCP**: Automatyczne testy integracyjne (tokio test) weryfikują pełen cykl: `index_document` → `semantic_search` → `delete_document` z co najmniej **5 różnymi dokumentami testowymi**.

5. **Dokumentacja**: README zawiera działające przykłady konfiguracji MCP, potwierdzone przez uruchomienie na czystej instalacji macOS ARM64.

---

## 6. Ryzyka projektowe

### R-01 — KD-Tree nieskuteczny dla 768-dim (Wysokie prawdopodobieństwo / Niski wpływ)
**Opis**: KD-Tree degeneruje do O(n) dla D=768 z powodu curse of dimensionality — odległość do płaszczyzny podziału jest porównywalna z odległością do najbliższego sąsiada.
**Mitygacja**: Zidentyfikowane z wyprzedzeniem. System automatycznie przełącza się na brute-force SIMD powyżej progu D=20. KD-Tree nadal wartościowy dla demonstracji algorytmu. Dodatkowy nakład pracy: 0 rh.

### R-02 — API `rmcp` niezgodne z kodem przykładowym (Średnie / Średni wpływ)
**Opis**: Crate `rmcp = "0.1"` może mieć API inne niż opisane w dokumentacji (wczesna wersja). Interfejsy `StdioServer`, `SseServer` mogą się różnić.
**Mitygacja**: Przed implementacją MCP — weryfikacja aktualnego API przez `cargo doc --open`. Jeśli API jest niekompletne: rozważenie bezpośredniej implementacji JSON-RPC 2.0 po `tokio::io`, co zwiększy nakład o ~20 rh.

### R-03 — Koszty OpenAI API podczas testów (Niskie / Niski wpływ)
**Opis**: Intensywne testy `index_document` z realnym API embeddingów mogą wygenerować nieoczekiwane koszty.
**Mitygacja**: Testy integracyjne używają `MockEmbedder` z deterministycznymi wektorami. Realne wywołania API tylko w testach demonstracyjnych. Koszt szacowany: < 5 USD przez cały projekt.

### R-04 — Niedostępność AVX2 na maszynie testowej (Niskie / Średni wpływ)
**Opis**: Maszyna bez AVX2 (stary CPU lub VM) nie wykona ścieżki AVX2, co uniemożliwi zmierzenie przyspieszenia.
**Mitygacja**: Warunkowa kompilacja z `#[cfg(target_feature = "avx2")]`. Fallback na NEON (ARM) lub scalar. CI używa `ubuntu-latest` (AVX2) i `macos-latest` (NEON). Benchmarki zawierają wyniki dla obu architektur.

### R-05 — Niestabilność `portable_simd` (nightly) (Niskie / Niski wpływ)
**Opis**: Feature `portable_simd` jest dostępna tylko na nightly Rust i może się zmienić między wersjami.
**Mitygacja**: Strategia hybrydowa: platform-specific intrinsics (`std::arch`) na stable jako główna ścieżka; portable SIMD jako opcja. Ryzyko zerowe dla deliverables na stable Rust.

### R-06 — Przekroczenie limitu RAM na maszynie demonstracyjnej (Niskie / Niski wpływ)
**Opis**: Demo z 1M wektorów 768-dim wymaga ~3,2 GB RAM, co może być problematyczne na słabszym sprzęcie.
**Mitygacja**: Zmniejszenie rozmiaru demo do 100K wektorów (320 MB). Wyniki skalują się liniowo — benchmark na 10K + 100K wystarczy do demonstracji wydajności.

---

*Dokument sporządzony: 2026-03-18 | Wersja: 1.0*
