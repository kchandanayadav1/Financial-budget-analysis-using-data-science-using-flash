"""
BudgetIQ - Financial Budget Analysis
Flask + Data Science Mini Project
Now with: History feature + CSV persistence
"""

from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import numpy as np
import io, base64, os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'budgetiq_2024'

# ── In-memory data store ──
db = {"income": [], "expenses": []}

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

CATEGORIES = ["Food & Dining","Housing","Transportation","Entertainment",
              "Healthcare","Education","Shopping","Utilities","Savings","Other"]

COLORS = ['#38BDF8','#818CF8','#34D399','#FB923C','#F472B6',
          '#A78BFA','#FBBF24','#4ADE80','#F87171','#67E8F9']

# ── CSV file paths ──
DATA_DIR     = os.path.join(os.path.dirname(__file__), 'data')
INCOME_CSV   = os.path.join(DATA_DIR, 'income.csv')
EXPENSES_CSV = os.path.join(DATA_DIR, 'expenses.csv')
HISTORY_CSV  = os.path.join(DATA_DIR, 'history.csv')
os.makedirs(DATA_DIR, exist_ok=True)


# ════════════════════════════════════════════════
# CSV PERSISTENCE
# ════════════════════════════════════════════════

def save_to_csv():
    """Save current db to CSV — data survives server restarts."""
    if db['income']:
        pd.DataFrame(db['income']).to_csv(INCOME_CSV, index=False)
    if db['expenses']:
        pd.DataFrame(db['expenses']).to_csv(EXPENSES_CSV, index=False)

def load_from_csv():
    """Load saved data from CSV files when server starts."""
    if os.path.exists(INCOME_CSV):
        try:
            df = pd.read_csv(INCOME_CSV).fillna('')
            db['income'] = df.to_dict('records')
            for i in db['income']:
                i['amount'] = float(i['amount'])
        except Exception:
            pass
    if os.path.exists(EXPENSES_CSV):
        try:
            df = pd.read_csv(EXPENSES_CSV).fillna('')
            db['expenses'] = df.to_dict('records')
            for e in db['expenses']:
                e['amount']   = float(e['amount'])
                e['budgeted'] = float(e['budgeted'])
        except Exception:
            pass

def save_month_to_history(month, year):
    """Save one month's complete budget snapshot into history.csv."""
    inc_rows = [i for i in db['income']   if i['month'] == month]
    exp_rows = [e for e in db['expenses'] if e['month'] == month]

    total_inc = sum(i['amount'] for i in inc_rows)
    total_exp = sum(e['amount'] for e in exp_rows)
    savings   = total_inc - total_exp
    sav_rate  = round(savings / total_inc * 100, 1) if total_inc > 0 else 0

    cat = {}
    for e in exp_rows:
        cat[e['category']] = cat.get(e['category'], 0) + e['amount']

    row = {
        'month':         month,
        'year':          year,
        'saved_on':      datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total_income':  round(total_inc, 2),
        'total_expense': round(total_exp, 2),
        'savings':       round(savings, 2),
        'savings_rate':  sav_rate,
        'inc_entries':   len(inc_rows),
        'exp_entries':   len(exp_rows),
    }
    for cat_name in CATEGORIES:
        key = cat_name.replace(' & ', '_').replace(' ', '_')
        row[key] = round(cat.get(cat_name, 0), 2)

    existing = []
    if os.path.exists(HISTORY_CSV):
        try:
            existing = pd.read_csv(HISTORY_CSV).fillna(0).to_dict('records')
        except Exception:
            existing = []

    # Overwrite same month+year if already saved
    existing = [r for r in existing
                if not (str(r.get('month')) == month and str(r.get('year')) == str(year))]
    existing.append(row)
    pd.DataFrame(existing).to_csv(HISTORY_CSV, index=False)
    return row

def load_history():
    """Load all saved history records, sorted by year → month."""
    if not os.path.exists(HISTORY_CSV):
        return []
    try:
        df = pd.read_csv(HISTORY_CSV).fillna(0)
        records = df.to_dict('records')
        records.sort(key=lambda r: (
            int(r.get('year', 2024)),
            MONTHS.index(r['month']) if r['month'] in MONTHS else 99
        ))
        return records
    except Exception:
        return []

# Load persisted data on startup
load_from_csv()


# ── Chart helpers ──
def to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0B0F1A', edgecolor='none', dpi=130)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def chart_style():
    plt.rcParams.update({
        'figure.facecolor':'#0B0F1A','axes.facecolor':'#131929',
        'axes.edgecolor':'#1E2D45','axes.labelcolor':'#94A3B8',
        'xtick.color':'#64748B','ytick.color':'#64748B',
        'text.color':'#E2E8F0','grid.color':'#1E2D45','grid.alpha':0.6,
    })

def empty_chart(msg):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor('#131929'); fig.patch.set_facecolor('#0B0F1A')
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            color='#475569', fontsize=11, transform=ax.transAxes, multialignment='center')
    ax.axis('off')
    return fig


# ════════════════════════════════════════════════
# PAGES
# ════════════════════════════════════════════════
@app.route('/')
def home():
    return render_template('index.html', categories=CATEGORIES, months=MONTHS)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
def history_page():
    return render_template('history.html', months=MONTHS)


# ════════════════════════════════════════════════
# API — DATA ENTRY
# ════════════════════════════════════════════════
@app.route('/api/add_income', methods=['POST'])
def add_income():
    d = request.json
    amt = float(d.get('amount', 0))
    if amt <= 0:
        return jsonify(success=False, message="Amount must be greater than 0")
    db['income'].append({
        "source": d.get('source','Salary').strip() or 'Salary',
        "amount": amt, "month": d.get('month','January'),
        "note":   d.get('note','').strip()
    })
    save_to_csv()
    total = sum(i['amount'] for i in db['income'])
    return jsonify(success=True, message=f"Income added! Total: ₹{total:,.0f}",
                   total_income=total, count=len(db['income']))

@app.route('/api/add_expense', methods=['POST'])
def add_expense():
    d = request.json
    amt = float(d.get('amount', 0))
    if amt <= 0:
        return jsonify(success=False, message="Amount must be greater than 0")
    db['expenses'].append({
        "category":    d.get('category','Other'),
        "amount":      amt,
        "budgeted":    float(d.get('budgeted', 0)),
        "month":       d.get('month','January'),
        "description": d.get('description','').strip()
    })
    save_to_csv()
    return jsonify(success=True, message="Expense added!", count=len(db['expenses']))


# ════════════════════════════════════════════════
# API — CSV UPLOAD
# ════════════════════════════════════════════════
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify(success=False, message="No file selected")
    try:
        df = pd.read_csv(request.files['file'])
        df.columns = df.columns.str.strip().str.lower()
        df = df.fillna('')
        mode = request.form.get('mode','replace')

        has_type_col     = 'type'     in df.columns
        has_expense_cols = 'amount'   in df.columns and 'category' in df.columns
        has_income_col   = 'income'   in df.columns

        if not has_type_col and not has_expense_cols and not has_income_col:
            return jsonify(success=False, message="CSV needs 'type' column OR 'category'+'amount' columns")

        exp_count = inc_count = skipped = 0

        if has_type_col:
            if mode == 'replace': db['expenses'].clear(); db['income'].clear()
            for _, row in df.iterrows():
                row_type = str(row.get('type','')).strip().lower()
                try:
                    if row_type == 'expense':
                        amt = float(row.get('amount', 0))
                        if amt <= 0: skipped += 1; continue
                        db['expenses'].append({
                            "category":    str(row.get('category','Other')).strip(),
                            "amount":      amt,
                            "budgeted":    float(row['budgeted']) if row.get('budgeted') != '' else 0,
                            "month":       str(row.get('month','January')).strip(),
                            "description": str(row.get('description','')).strip()
                        }); exp_count += 1
                    elif row_type == 'income':
                        amt = float(row.get('amount', 0))
                        if amt <= 0: skipped += 1; continue
                        db['income'].append({
                            "source": str(row.get('category','Salary')).strip() or 'Salary',
                            "amount": amt,
                            "month":  str(row.get('month','January')).strip(),
                            "note":   str(row.get('description','')).strip()
                        }); inc_count += 1
                    else: skipped += 1
                except (ValueError, TypeError): skipped += 1
        else:
            if has_expense_cols and mode == 'replace': db['expenses'].clear()
            if has_income_col   and mode == 'replace': db['income'].clear()
            for _, row in df.iterrows():
                if has_expense_cols:
                    try:
                        amt = float(row['amount'])
                        if amt <= 0: skipped += 1; continue
                        db['expenses'].append({
                            "category":    str(row.get('category','Other')).strip(),
                            "amount":      amt,
                            "budgeted":    float(row['budgeted']) if row.get('budgeted') != '' else 0,
                            "month":       str(row.get('month','January')).strip(),
                            "description": str(row.get('description','')).strip()
                        }); exp_count += 1
                    except (ValueError, TypeError): skipped += 1
                if has_income_col:
                    try:
                        inc = float(row['income'])
                        if inc > 0:
                            db['income'].append({
                                "source": str(row.get('source','Salary')).strip() or 'Salary',
                                "amount": inc,
                                "month":  str(row.get('month','January')).strip(),
                                "note":   "Imported from CSV"
                            }); inc_count += 1
                    except (ValueError, TypeError): pass

        save_to_csv()
        parts = []
        if exp_count: parts.append(f"{exp_count} expenses")
        if inc_count: parts.append(f"{inc_count} income entries")
        if not parts: parts.append("0 records")
        skip_note = f" ({skipped} rows skipped)" if skipped else ""
        return jsonify(success=True,
                       message=f"Imported {' & '.join(parts)}{skip_note}",
                       exp_count=exp_count, inc_count=inc_count, columns=list(df.columns))
    except Exception as e:
        return jsonify(success=False, message=f"Error reading CSV: {str(e)}")


# ════════════════════════════════════════════════
# API — SUMMARY & DATA
# ════════════════════════════════════════════════
@app.route('/api/summary')
def summary():
    total_inc = sum(i['amount'] for i in db['income'])
    total_exp = sum(e['amount'] for e in db['expenses'])
    savings   = total_inc - total_exp
    sav_rate  = round(savings / total_inc * 100, 1) if total_inc > 0 else 0
    cat = {}
    for e in db['expenses']:
        cat[e['category']] = cat.get(e['category'], 0) + e['amount']
    mon_exp = {}; mon_inc = {}
    for e in db['expenses']:
        mon_exp[e['month']] = mon_exp.get(e['month'], 0) + e['amount']
    for i in db['income']:
        mon_inc[i['month']] = mon_inc.get(i['month'], 0) + i['amount']
    return jsonify(total_income=round(total_inc,2), total_expenses=round(total_exp,2),
                   savings=round(savings,2), savings_rate=sav_rate,
                   income_count=len(db['income']), expense_count=len(db['expenses']),
                   category_totals=cat, monthly_exp=mon_exp, monthly_inc=mon_inc)

@app.route('/api/data')
def get_data():
    return jsonify(income=db['income'], expenses=db['expenses'])

@app.route('/api/reset', methods=['POST'])
def reset():
    db['income'].clear(); db['expenses'].clear()
    for f in [INCOME_CSV, EXPENSES_CSV]:
        if os.path.exists(f): os.remove(f)
    return jsonify(success=True, message="All data cleared!")


# ════════════════════════════════════════════════
# API — EDIT & DELETE ENTRY (NEW)
# ════════════════════════════════════════════════

@app.route('/api/edit_entry', methods=['POST'])
def edit_entry():
    """Edit an existing income or expense entry by index."""
    d     = request.json
    etype = d.get('type', '')     # 'expense' or 'income'
    idx   = d.get('index', -1)

    if etype == 'expense':
        if idx < 0 or idx >= len(db['expenses']):
            return jsonify(success=False, message="Entry not found")
        amt = float(d.get('amount', 0))
        if amt <= 0:
            return jsonify(success=False, message="Amount must be greater than 0")
        db['expenses'][idx] = {
            'category':    d.get('category', 'Other'),
            'amount':      amt,
            'budgeted':    float(d.get('budgeted', 0)),
            'month':       d.get('month', 'January'),
            'description': d.get('description', '').strip()
        }
        save_to_csv()
        return jsonify(success=True, message=f"Expense updated!")

    elif etype == 'income':
        if idx < 0 or idx >= len(db['income']):
            return jsonify(success=False, message="Entry not found")
        amt = float(d.get('amount', 0))
        if amt <= 0:
            return jsonify(success=False, message="Amount must be greater than 0")
        db['income'][idx] = {
            'source': d.get('source', 'Salary').strip() or 'Salary',
            'amount': amt,
            'month':  d.get('month', 'January'),
            'note':   d.get('note', '').strip()
        }
        save_to_csv()
        return jsonify(success=True, message="Income updated!")

    return jsonify(success=False, message="Invalid type")


@app.route('/api/delete_entry', methods=['POST'])
def delete_entry():
    """Delete an income or expense entry by index."""
    d     = request.json
    etype = d.get('type', '')
    idx   = d.get('index', -1)

    if etype == 'expense':
        if idx < 0 or idx >= len(db['expenses']):
            return jsonify(success=False, message="Entry not found")
        removed = db['expenses'].pop(idx)
        save_to_csv()
        return jsonify(success=True, message=f"{removed['category']} expense deleted")

    elif etype == 'income':
        if idx < 0 or idx >= len(db['income']):
            return jsonify(success=False, message="Entry not found")
        removed = db['income'].pop(idx)
        save_to_csv()
        return jsonify(success=True, message=f"{removed['source']} income deleted")

    return jsonify(success=False, message="Invalid type")


# ════════════════════════════════════════════════
# API — DOWNLOAD CSV
# ════════════════════════════════════════════════

@app.route('/api/download_csv')
def download_csv():
    """
    Download income + expenses as CSV.
    Optional ?month=January filter to download a single month only.
    """
    month_filter = request.args.get('month', '').strip()  # e.g. ?month=January
    filter_active = month_filter in MONTHS

    # Pick which entries to include
    income_data   = [i for i in db['income']   if not filter_active or i.get('month') == month_filter]
    expenses_data = [e for e in db['expenses'] if not filter_active or e.get('month') == month_filter]

    if not income_data and not expenses_data:
        msg = f'No data found for {month_filter}.' if filter_active else 'No data to download.'
        # Return empty CSV with just header
        csv_content = 'type,category,amount,budgeted,month,description\n'
        filename = f"budgetiq_{month_filter or 'empty'}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        return Response(csv_content, mimetype='text/csv',
                        headers={'Content-Disposition': f'attachment; filename={filename}'})

    rows = []
    for i in income_data:
        rows.append({
            'type':        'income',
            'category':    i.get('source', 'Salary'),
            'amount':      i.get('amount', 0),
            'budgeted':    '',
            'month':       i.get('month', ''),
            'description': i.get('note', ''),
        })
    for e in expenses_data:
        rows.append({
            'type':        'expense',
            'category':    e.get('category', ''),
            'amount':      e.get('amount', 0),
            'budgeted':    e.get('budgeted', 0),
            'month':       e.get('month', ''),
            'description': e.get('description', ''),
        })

    df = pd.DataFrame(rows, columns=['type','category','amount','budgeted','month','description'])
    df['_order'] = df['month'].apply(lambda m: MONTHS.index(m) if m in MONTHS else 99)
    df = df.sort_values(['_order', 'type']).drop(columns='_order')
    csv_content = df.to_csv(index=False)

    # Filename includes month name if filtered
    if filter_active:
        filename = f"budgetiq_{month_filter}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    else:
        filename = f"budgetiq_all_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

    return Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )

@app.route('/api/history/save', methods=['POST'])
def history_save():
    """Save a month's snapshot to history.csv permanently."""
    d     = request.json
    month = d.get('month', '')
    year  = d.get('year', datetime.now().year)

    if month not in MONTHS:
        return jsonify(success=False, message="Invalid month selected")

    inc_rows = [i for i in db['income']   if i['month'] == month]
    exp_rows = [e for e in db['expenses'] if e['month'] == month]

    if not inc_rows and not exp_rows:
        return jsonify(success=False,
                       message=f"No data found for {month}. Add income/expenses first!")

    row = save_month_to_history(month, year)
    return jsonify(success=True,
                   message=f"✅ {month} {year} saved to history!",
                   snapshot=row)


@app.route('/api/history/list')
def history_list():
    """Return all saved monthly history records."""
    records = load_history()
    return jsonify(history=records, count=len(records))


@app.route('/api/history/delete', methods=['POST'])
def history_delete():
    """Delete one month from history."""
    d     = request.json
    month = d.get('month', '')
    year  = str(d.get('year', ''))

    if not os.path.exists(HISTORY_CSV):
        return jsonify(success=False, message="No history found")

    existing = pd.read_csv(HISTORY_CSV).fillna(0).to_dict('records')
    filtered = [r for r in existing
                if not (str(r.get('month')) == month and str(r.get('year')) == year)]

    if len(filtered) == len(existing):
        return jsonify(success=False, message="Record not found")

    pd.DataFrame(filtered).to_csv(HISTORY_CSV, index=False)
    return jsonify(success=True, message=f"🗑 {month} {year} deleted from history")


@app.route('/api/history/month_summary')
def history_month_summary():
    """Return summary for a specific month from current db."""
    month = request.args.get('month', '')
    if month not in MONTHS:
        return jsonify(success=False, message="Invalid month")

    inc_rows = [i for i in db['income']   if i['month'] == month]
    exp_rows = [e for e in db['expenses'] if e['month'] == month]

    total_inc = sum(i['amount'] for i in inc_rows)
    total_exp = sum(e['amount'] for e in exp_rows)
    savings   = total_inc - total_exp
    sav_rate  = round(savings / total_inc * 100, 1) if total_inc > 0 else 0

    cat = {}
    for e in exp_rows:
        cat[e['category']] = cat.get(e['category'], 0) + e['amount']

    return jsonify(month=month, total_income=round(total_inc,2),
                   total_expenses=round(total_exp,2), savings=round(savings,2),
                   savings_rate=sav_rate, income_count=len(inc_rows),
                   expense_count=len(exp_rows), category_totals=cat,
                   expenses=exp_rows, income=inc_rows)


@app.route('/api/history/compare')
def history_compare():
    """Compare two saved months side by side."""
    m1 = request.args.get('month1','')
    m2 = request.args.get('month2','')
    y1 = request.args.get('year1', str(datetime.now().year))
    y2 = request.args.get('year2', str(datetime.now().year))

    records = load_history()
    def find(month, year):
        for r in records:
            if str(r.get('month')) == month and str(r.get('year')) == str(year):
                return r
        return None

    r1 = find(m1, y1)
    r2 = find(m2, y2)

    if not r1: return jsonify(success=False, message=f"{m1} {y1} not in history. Save it first!")
    if not r2: return jsonify(success=False, message=f"{m2} {y2} not in history. Save it first!")
    return jsonify(success=True, month1=r1, month2=r2)


# ════════════════════════════════════════════════
# API — HISTORY CHARTS
# ════════════════════════════════════════════════

@app.route('/api/history/chart/categories')
def history_chart_categories():
    chart_style()
    month    = request.args.get('month','')
    exp_rows = [e for e in db['expenses'] if e['month'] == month]
    cat = {}
    for e in exp_rows:
        cat[e['category']] = cat.get(e['category'], 0) + e['amount']
    if not cat:
        return jsonify(chart=to_base64(empty_chart(f'No expenses for {month}')))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    wedges, texts, autos = ax.pie(
        cat.values(), labels=cat.keys(), autopct='%1.1f%%',
        colors=COLORS[:len(cat)], pctdistance=0.82,
        wedgeprops=dict(width=0.55, edgecolor='#0B0F1A', linewidth=2))
    for t in texts:  t.set_color('#94A3B8'); t.set_fontsize(7.5)
    for a in autos:  a.set_color('#0B0F1A'); a.set_fontsize(7); a.set_fontweight('bold')
    ax.set_title(f'{month} — Spending by Category', color='#38BDF8', fontsize=11, fontweight='bold')
    return jsonify(chart=to_base64(fig))


@app.route('/api/history/chart/bva')
def history_chart_bva():
    chart_style()
    month    = request.args.get('month','')
    exp_rows = [e for e in db['expenses'] if e['month'] == month]
    cat_actual = {}; cat_budget = {}
    for e in exp_rows:
        c = e['category']
        cat_actual[c] = cat_actual.get(c,0) + e['amount']
        if e['budgeted'] > 0:
            cat_budget[c] = cat_budget.get(c,0) + e['budgeted']
    if not cat_actual:
        return jsonify(chart=to_base64(empty_chart(f'No expense data for {month}')))
    cats    = list(cat_actual.keys())
    actuals = [cat_actual[c] for c in cats]
    budgets = [cat_budget.get(c,0) for c in cats]
    x = np.arange(len(cats)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x-w/2, budgets, w, label='Budgeted', color='#38BDF8', alpha=0.8, edgecolor='#0B0F1A')
    b2 = ax.bar(x+w/2, actuals, w, label='Actual',   color='#818CF8', alpha=0.8, edgecolor='#0B0F1A')
    for i,(bud,act) in enumerate(zip(budgets,actuals)):
        if bud > 0 and act > bud:
            b2[i].set_color('#F87171'); b2[i].set_alpha(1.0)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=25, ha='right', fontsize=8)
    ax.set_title(f'{month} — Budget vs Actual', color='#38BDF8', fontsize=11, fontweight='bold')
    ax.legend(facecolor='#131929', edgecolor='#1E2D45', labelcolor='#E2E8F0', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'₹{v:,.0f}'))
    return jsonify(chart=to_base64(fig))


@app.route('/api/history/chart/compare')
def history_chart_compare():
    chart_style()
    m1 = request.args.get('month1',''); m2 = request.args.get('month2','')
    y1 = request.args.get('year1', str(datetime.now().year))
    y2 = request.args.get('year2', str(datetime.now().year))
    records = load_history()
    def find(m, y):
        for r in records:
            if str(r.get('month'))==m and str(r.get('year'))==str(y): return r
        return None
    r1 = find(m1,y1); r2 = find(m2,y2)
    if not r1 or not r2:
        return jsonify(chart=to_base64(empty_chart('Save both months to history\nbefore comparing')))

    labels = ['Income','Expenses','Savings']
    vals1  = [float(r1.get('total_income',0)), float(r1.get('total_expense',0)), float(r1.get('savings',0))]
    vals2  = [float(r2.get('total_income',0)), float(r2.get('total_expense',0)), float(r2.get('savings',0))]
    x = np.arange(len(labels)); w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(x-w/2, vals1, w, label=f'{m1} {y1}', color='#38BDF8', alpha=0.85, edgecolor='#0B0F1A')
    ax1.bar(x+w/2, vals2, w, label=f'{m2} {y2}', color='#F472B6', alpha=0.85, edgecolor='#0B0F1A')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_title('Income / Expenses / Savings', color='#38BDF8', fontsize=11, fontweight='bold')
    ax1.legend(facecolor='#131929', edgecolor='#1E2D45', labelcolor='#E2E8F0', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'₹{v:,.0f}'))

    cat_cols   = [c.replace(' & ','_').replace(' ','_') for c in CATEGORIES]
    visible    = [(CATEGORIES[i], float(r1.get(cat_cols[i],0)), float(r2.get(cat_cols[i],0)))
                  for i in range(len(CATEGORIES))
                  if float(r1.get(cat_cols[i],0))>0 or float(r2.get(cat_cols[i],0))>0]
    if visible:
        vl,vv1,vv2 = zip(*visible)
        xc = np.arange(len(vl)); wc = 0.35
        ax2.bar(xc-wc/2, vv1, wc, label=f'{m1} {y1}', color='#38BDF8', alpha=0.85, edgecolor='#0B0F1A')
        ax2.bar(xc+wc/2, vv2, wc, label=f'{m2} {y2}', color='#F472B6', alpha=0.85, edgecolor='#0B0F1A')
        ax2.set_xticks(xc); ax2.set_xticklabels(vl, rotation=30, ha='right', fontsize=8)
        ax2.legend(facecolor='#131929', edgecolor='#1E2D45', labelcolor='#E2E8F0', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'₹{v:,.0f}'))
    ax2.set_title('Category Comparison', color='#38BDF8', fontsize=11, fontweight='bold')
    fig.suptitle(f'Comparing  {m1} {y1}  vs  {m2} {y2}',
                 color='#E2E8F0', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return jsonify(chart=to_base64(fig))


# ════════════════════════════════════════════════
# API — DASHBOARD CHARTS (unchanged)
# ════════════════════════════════════════════════
@app.route('/api/chart/categories')
def chart_categories():
    chart_style()
    cat = {}
    for e in db['expenses']:
        cat[e['category']] = cat.get(e['category'], 0) + e['amount']
    if not cat:
        return jsonify(chart=to_base64(empty_chart('Add expenses to see\ncategory breakdown')))
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autos = ax.pie(
        cat.values(), labels=cat.keys(), autopct='%1.1f%%',
        colors=COLORS[:len(cat)], pctdistance=0.82,
        wedgeprops=dict(width=0.55, edgecolor='#0B0F1A', linewidth=2.5))
    for t in texts:  t.set_color('#94A3B8'); t.set_fontsize(8)
    for a in autos:  a.set_color('#0B0F1A'); a.set_fontsize(7); a.set_fontweight('bold')
    ax.set_title('Spending by Category', color='#38BDF8', fontsize=13, fontweight='bold', pad=12)
    return jsonify(chart=to_base64(fig))

@app.route('/api/chart/budget_vs_actual')
def chart_bva():
    chart_style()
    cat_actual = {}; cat_budget = {}
    for e in db['expenses']:
        c = e['category']
        cat_actual[c] = cat_actual.get(c,0) + e['amount']
        if e['budgeted'] > 0:
            cat_budget[c] = cat_budget.get(c,0) + e['budgeted']
    if not cat_actual:
        return jsonify(chart=to_base64(empty_chart('Add expenses with budget\namounts to see comparison')))
    cats    = list(cat_actual.keys())
    actuals = [cat_actual[c] for c in cats]
    budgets = [cat_budget.get(c,0) for c in cats]
    x = np.arange(len(cats)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x-w/2, budgets, w, label='Budgeted', color='#38BDF8', alpha=0.8, edgecolor='#0B0F1A')
    b2 = ax.bar(x+w/2, actuals, w, label='Actual',   color='#818CF8', alpha=0.8, edgecolor='#0B0F1A')
    for i,(bud,act) in enumerate(zip(budgets,actuals)):
        if bud > 0 and act > bud:
            b2[i].set_color('#F87171'); b2[i].set_alpha(1.0)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=25, ha='right', fontsize=8)
    ax.set_title('Budget vs Actual Spending', color='#38BDF8', fontsize=13, fontweight='bold')
    ax.legend(facecolor='#131929', edgecolor='#1E2D45', labelcolor='#E2E8F0', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'₹{v:,.0f}'))
    return jsonify(chart=to_base64(fig))

@app.route('/api/chart/monthly_trend')
def chart_monthly():
    chart_style()
    mon_exp = {}; mon_inc = {}
    for e in db['expenses']:
        mon_exp[e['month']] = mon_exp.get(e['month'], 0) + e['amount']
    for i in db['income']:
        mon_inc[i['month']] = mon_inc.get(i['month'], 0) + i['amount']
    all_months = sorted(set(list(mon_exp)+list(mon_inc)),
                        key=lambda m: MONTHS.index(m) if m in MONTHS else 99)
    if not all_months:
        return jsonify(chart=to_base64(empty_chart('Add data across multiple\nmonths to see trend')))
    inc_vals = [mon_inc.get(m,0) for m in all_months]
    exp_vals = [mon_exp.get(m,0) for m in all_months]
    sav_vals = [inc_vals[i]-exp_vals[i] for i in range(len(all_months))]
    x = np.arange(len(all_months))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x, inc_vals, alpha=0.08, color='#34D399')
    ax.fill_between(x, exp_vals, alpha=0.08, color='#F472B6')
    ax.plot(x, inc_vals, 'o-', color='#34D399', lw=2.5, ms=7, label='Income')
    ax.plot(x, exp_vals, 'o-', color='#F472B6', lw=2.5, ms=7, label='Expenses')
    ax.plot(x, sav_vals, 's--',color='#FBBF24', lw=2,   ms=6, label='Savings')
    ax.set_xticks(x); ax.set_xticklabels([m[:3] for m in all_months], fontsize=9)
    ax.set_title('Monthly Income vs Expenses vs Savings', color='#38BDF8', fontsize=13, fontweight='bold')
    ax.legend(facecolor='#131929', edgecolor='#1E2D45', labelcolor='#E2E8F0', fontsize=9)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'₹{v:,.0f}'))
    return jsonify(chart=to_base64(fig))

@app.route('/api/chart/forecast')
def chart_forecast():
    chart_style()
    mon_exp = {}
    for e in db['expenses']:
        mon_exp[e['month']] = mon_exp.get(e['month'], 0) + e['amount']
    sorted_months = sorted(mon_exp.keys(),
                           key=lambda m: MONTHS.index(m) if m in MONTHS else 99)
    if len(sorted_months) < 2:
        return jsonify(chart=to_base64(
            empty_chart('Need at least 2 months\nof data for ML forecasting')), message="Need more data")
    y = np.array([mon_exp[m] for m in sorted_months])
    X = np.arange(len(y)).reshape(-1,1)
    deg = min(2, len(y)-1)
    poly = PolynomialFeatures(degree=deg)
    Xp   = poly.fit_transform(X)
    model= LinearRegression().fit(Xp,y)
    fut_X= np.arange(len(y),len(y)+3).reshape(-1,1)
    fut_y= np.maximum(model.predict(poly.transform(fut_X)),0)
    r2   = max(model.score(Xp,y),0)
    fig,ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(sorted_months))
    ax.bar(x,y,color='#38BDF8',alpha=0.75,edgecolor='#0B0F1A',label='Actual',width=0.6)
    xl = np.linspace(0,len(y)-1,200).reshape(-1,1)
    ax.plot(xl,model.predict(poly.transform(xl)),'--',color='#34D399',lw=2,alpha=0.8,label='Trend')
    fx = np.arange(len(y),len(y)+3)
    ax.bar(fx,fut_y,color='#F472B6',alpha=0.65,edgecolor='#0B0F1A',label='Forecast',width=0.6)
    ax.axvspan(len(y)-0.5,len(y)+2.5,alpha=0.04,color='#F472B6')
    ax.axvline(len(y)-0.5,color='#FBBF24',lw=1.5,ls=':',alpha=0.7)
    ax.text(len(y)+0.1,max(y)*0.97,'← Forecast',color='#FBBF24',fontsize=8)
    labels=[m[:3] for m in sorted_months]+[f'+{i+1}mo' for i in range(3)]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,fontsize=8)
    ax.set_title(f'Expense Forecast — Polynomial Regression  (R² = {r2:.3f})',
                 color='#38BDF8',fontsize=12,fontweight='bold')
    ax.legend(facecolor='#131929',edgecolor='#1E2D45',labelcolor='#E2E8F0',fontsize=9)
    ax.grid(axis='y',alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'₹{v:,.0f}'))
    return jsonify(chart=to_base64(fig),
                   forecast=[round(float(v),2) for v in fut_y], r2=round(float(r2),3))


# ════════════════════════════════════════════════
# API — AI SUGGESTIONS
# ════════════════════════════════════════════════
@app.route('/api/suggestions')
def suggestions():
    total_inc = sum(i['amount'] for i in db['income'])
    total_exp = sum(e['amount'] for e in db['expenses'])
    savings   = total_inc - total_exp
    sav_rate  = (savings/total_inc*100) if total_inc > 0 else 0
    if total_inc == 0:
        return jsonify(alerts=[], tips=["Add your income first to get AI suggestions!"],
                       wins=[], score=0, grade='—', rule={"needs":0,"wants":0,"savings":0})
    cat = {}
    for e in db['expenses']:
        cat[e['category']] = cat.get(e['category'],0) + e['amount']
    needs = (cat.get('Food & Dining',0)+cat.get('Housing',0)+cat.get('Utilities',0)
             +cat.get('Healthcare',0)+cat.get('Transportation',0)) / total_inc * 100
    wants = (cat.get('Entertainment',0)+cat.get('Shopping',0)) / total_inc * 100
    alerts,tips,wins=[],[],[]
    score=100
    if sav_rate<0:    alerts.append(f"🚨 Spending ₹{abs(savings):,.0f} MORE than income!"); score-=35
    elif sav_rate<10: alerts.append(f"⚠️ Savings rate only {sav_rate:.1f}%. Aim for 20%."); score-=20
    elif sav_rate>=20:wins.append(f"✅ Excellent savings rate of {sav_rate:.1f}%!")
    if needs>50: alerts.append(f"🏠 Essential spending ({needs:.1f}%) exceeds 50%."); score-=10
    if wants>30: alerts.append(f"🛍️ Lifestyle spending ({wants:.1f}%) exceeds 30%."); score-=10
    food=cat.get('Food & Dining',0)
    if food>total_inc*0.25: tips.append(f"🍔 Food (₹{food:,.0f}) is high. Meal prep saves 30-40%.")
    if db['expenses']:
        mc=len(set(e['month'] for e in db['expenses'])) or 1
        tips.append(f"💡 Emergency fund target: ₹{total_exp/mc*6:,.0f}")
    if cat:
        top=max(cat,key=cat.get)
        tips.append(f"📌 Biggest expense: '{top}' at ₹{cat[top]:,.0f}")
    if savings>0: wins.append(f"💰 Total saved: ₹{savings:,.0f}. Keep it up!")
    score=max(0,min(100,score))
    grade='A' if score>=90 else 'B' if score>=75 else 'C' if score>=60 else 'D' if score>=40 else 'F'
    return jsonify(alerts=alerts,tips=tips,wins=wins,score=score,grade=grade,
                   rule={"needs":round(needs,1),"wants":round(wants,1),"savings":round(sav_rate,1)})


# ════════════════════════════════════════════════
# API — LOAD SAMPLE DATA
# ════════════════════════════════════════════════
@app.route('/api/load_sample', methods=['POST'])
def load_sample():
    db['income'].clear(); db['expenses'].clear()
    for month,amt in [("January",45000),("February",45000),("March",47000),
                      ("April",48000),("May",50000),("June",48000)]:
        db['income'].append({"source":"Salary","amount":amt,"month":month,"note":"Monthly salary"})
    sample=[
        ("Food & Dining",8500,10000,"January"),("Housing",12000,12000,"January"),
        ("Transportation",3200,3000,"January"),("Entertainment",4500,3000,"January"),
        ("Utilities",2100,2000,"January"),("Healthcare",1500,2000,"January"),
        ("Food & Dining",9200,10000,"February"),("Housing",12000,12000,"February"),
        ("Transportation",2800,3000,"February"),("Entertainment",3100,3000,"February"),
        ("Shopping",5600,4000,"February"),("Utilities",1900,2000,"February"),
        ("Food & Dining",7800,10000,"March"),("Housing",12000,12000,"March"),
        ("Transportation",3500,3000,"March"),("Entertainment",2200,3000,"March"),
        ("Education",4000,4000,"March"),("Utilities",2200,2000,"March"),
        ("Food & Dining",8100,10000,"April"),("Housing",12000,12000,"April"),
        ("Transportation",2900,3000,"April"),("Entertainment",3800,3000,"April"),
        ("Shopping",3200,4000,"April"),("Utilities",2000,2000,"April"),
        ("Food & Dining",9500,10000,"May"),("Housing",12000,12000,"May"),
        ("Transportation",3100,3000,"May"),("Entertainment",5200,3000,"May"),
        ("Healthcare",800,2000,"May"),("Utilities",2300,2000,"May"),
        ("Food & Dining",8800,10000,"June"),("Housing",12000,12000,"June"),
        ("Transportation",3000,3000,"June"),("Entertainment",2900,3000,"June"),
        ("Savings",5000,5000,"June"),("Utilities",2100,2000,"June"),
    ]
    for cat,amt,bud,month in sample:
        db['expenses'].append({"category":cat,"amount":amt,"budgeted":bud,"month":month,"description":""})
    save_to_csv()
    return jsonify(success=True, message="Sample data loaded!")

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)