# DAM integration #

We're creating a numerical solver for our Differential approximation model (DAM) of the Majda-McLaughlin-Tabak (MMT) model. The differential model focuses on a low order approximation of the MMT flux, by constructing a sort of parabolic model built on the spectral power *p*.

With this numerical integration we're trying to evolve the obtained system in order to reproduce some notable cases such as the direct energy cascade that leads the action spectral density $n_k$ to the known **Kolmogorov-Zakharov** solution 

## Code structure 

the structure is pretty simple:
 - **DAM.py** is a library containing all of the integration methods, the derivatives construction, the right hand side of the equation, boundary and initial conditions and the integration cycle, also a plotting function has been implemented.
 - **main_file.py** the main file is the one managing the integration procedure.

---

## Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy scipy matplotlib scikit-learn
pip install -r requirements.txt
```

Developed and tested on Ubuntu (WSL2), CPU-only (no CUDA-capable GPU available).

---

## Usage

```bash
#runmanager file to run the numerical integration
python main_DAM.py

```
The generated figures are saved in to `outputs/`