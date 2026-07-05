import os
import re
import sqlite3

import streamlit as st
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

DB_FILE = "sample.db"
llm_model = "gemma-4-31b-it"
#llm_model = "gemini-2.5-flash"

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif os.getenv("GOOGLE_API_KEY"):
    pass
else:
    st.error("GOOGLE_API_KEY belum diset di Streamlit Secrets.")
    st.stop()


# -----------------------------
# DB helpers
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def get_sample_users():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT cif, name, phone
            FROM people
            ORDER BY cif
        """).fetchall()
        return [dict(row) for row in rows]


def verify_login(phone: str, pin: str):
    with get_conn() as conn:
        row = conn.execute("""
            SELECT person_id, cif, name, phone
            FROM people
            WHERE phone = ? AND pin = ? AND membership_status = 'ACTIVE'
            LIMIT 1
        """, (phone, pin)).fetchone()

        return dict(row) if row else None


# -----------------------------
# LangChain tools
# -----------------------------
@tool
def list_tables() -> list[str]:
    """Retrieve the names of all user tables in the SQLite database."""
    print(" - DB CALL: list_tables")

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """)
        return [row[0] for row in cursor.fetchall()]


@tool
def describe_table(table_name: str) -> list[tuple[str, str]]:
    """Look up one SQLite table schema.

    Returns:
      List of columns, where each entry is a tuple of (column, type).
    """
    print(f" - DB CALL: describe_table: {table_name}")

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
        raise ValueError("Invalid table name.")

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?;
        """, (table_name,))

        if cursor.fetchone() is None:
            return []

        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()

        return [(col[1], col[2]) for col in schema]


@tool
def execute_query(sql: str) -> list[dict]:
    """Execute a read-only SELECT statement and return rows as dictionaries."""
    print(f" - DB CALL: execute_query: {sql}")

    cleaned = sql.strip().lower()

    forbidden_keywords = [
        "insert", "update", "delete", "drop", "alter", "create",
        "replace", "truncate", "attach", "detach", "pragma"
    ]

    if not cleaned.startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")

    if any(keyword in cleaned for keyword in forbidden_keywords):
        raise ValueError("Query contains a forbidden keyword.")

    with get_conn() as conn:
        rows = conn.execute(sql).fetchall()
        return [dict(row) for row in rows]

def extract_text_content(content) -> str:
    """Extract visible text from model responses that may include content blocks."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts).strip()

    return str(content)

@st.cache_resource
def build_agent():
    instruction = """
You are a helpful Islamic Core Banking chatbot for members.

You can answer questions by querying a local SQLite database using the available tools.

Database domain:
- people: member identity, CIF, phone, PIN
- saving_account_types: available saving products and their differences
- saving_accounts: member saving accounts and balances
- saving_transactions: saving account transaction history
- financing_product_types: available financing products
- financing_accounts: member financing accounts

Critical identity rule:
- The user is already logged in.
- For any account-specific question, ALWAYS filter by the exact logged-in person_id.
- Do NOT search by partial name such as Ahmad.
- Do NOT answer that data is missing before querying with person_id.
- Never show or mention PIN values.

SQL workflow rules:
- ALWAYS start by calling list_tables.
- ALWAYS call describe_table on each relevant table before writing SQL.
- Never assume table names or column names.
- Only use execute_query for SELECT queries.
- When listing recent transactions, return maximum 5 transactions.
- When explaining saving products, explain only one saving product at a time.
- If the user asks broadly about saving products, list product names and ask which one they want explained.
- For financing simulation, calculate a simple monthly installment and return a markdown table.
- Answer in Indonesian.
"""

    model = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=0,
    )

    return create_agent(
        model=model,
        tools=[list_tables, describe_table, execute_query],
        system_prompt=instruction,
    )


def ask_agent(prompt: str, user: dict, history: list[dict]) -> str:
    context_prompt = f"""
Logged-in member:
person_id = "{user["person_id"]}"
cif = "{user["cif"]}"
name = "{user["name"]}"
phone = "{user["phone"]}"

Important:
For balance, saving accounts, financing accounts, and transactions,
you MUST query using:
WHERE person_id = "{user["person_id"]}"

User question:
{prompt}
"""

    messages = []

    # Jangan masukkan pesan user terakhir dua kali.
    for msg in history[-10:]:
        if msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "user" and msg["content"] != prompt:
            messages.append({"role": "user", "content": msg["content"]})

    messages.append({"role": "user", "content": context_prompt})

    agent = build_agent()
    response = agent.invoke({"messages": messages})
    return extract_text_content(response["messages"][-1].content)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Banking Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("Banking Chatbot")
st.caption("Chatbot simpanan dan pembiayaan berbasis SQLite + Gemini")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY belum tersedia. Set API key dulu sebelum menjalankan Streamlit.")
    st.stop()

if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.subheader("Sample Login")
    st.caption("Semua PIN demo: 123456")

    try:
        users = get_sample_users()
        for user in users:
            st.code(user["phone"])
    except Exception as e:
        st.error(f"Gagal membaca sample user: {e}")

    st.divider()
    st.subheader("Contoh Prompt")
    example_prompts = [
        "Mohon tampilkan saldo semua simpanan dalam tabel.",
        "Tampilkan 5 transaksi terakhir saya.",
        "Jenis simpanan apa saja yang tersedia?",
        "Jelaskan Simpanan Wadiah.",
        "Jenis pembiayaan apa saja yang tersedia?",
        "Simulasikan pembiayaan murabahah 12 juta tenor 24 bulan margin 12%.",
    ]
    for example_prompt in example_prompts:
        st.markdown(f"- {example_prompt}")

    st.divider()

    if st.session_state.user:
        st.success(f'Login: {st.session_state.user["name"]}')
        st.caption(f'CIF: {st.session_state.user["cif"]}')

        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()

        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.rerun()


if not st.session_state.user:
    st.subheader("Login Anggota")

    with st.form("login_form"):
        phone = st.text_input("No HP", placeholder="081234567001")
        pin = st.text_input("PIN", type="password", placeholder="123456")
        submitted = st.form_submit_button("Masuk")

    if submitted:
        user = verify_login(phone.strip(), pin.strip())
        if user:
            st.session_state.user = user
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"Halo {user['name']}. Saya bisa bantu cek saldo simpanan, transaksi terakhir, produk simpanan, produk pembiayaan, dan simulasi pembiayaan.",
                }
            ]
            st.rerun()
        else:
            st.error("No HP atau PIN tidak sesuai.")

    st.stop()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Tanyakan saldo, transaksi terakhir, produk simpanan, atau simulasi pembiayaan...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mengambil data..."):
            try:
                answer = ask_agent(
                    prompt=prompt,
                    user=st.session_state.user,
                    history=st.session_state.messages,
                )
            except Exception as e:
                answer = f"Terjadi error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
