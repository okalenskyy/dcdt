import io
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Defaults & Helpers
# -----------------------------

def get_default_products():
    data = [
        {
            "Product_ID": 1,
            "Product_Name": "Bare-Metal GPU Node",
            "Product_Type": "BareMetal",
            "GPU_SKU_Name": "NVIDIA_A100_80GB",
            "Price_per_GPUh_EUR": 0.45,
            "Variable_Cost_per_GPUh_EUR": 0.10,   # excl. power & generic var cost
            "Support_Cost_pct_of_Revenue": 0.03,
            "Utilization_Multiplier": 1.00,
            "Mix_Share_of_Total_Demand": 0.40,
        },
        {
            "Product_ID": 2,
            "Product_Name": "NeoCloud GPU Instance",
            "Product_Type": "Virtualized",
            "GPU_SKU_Name": "NVIDIA_H100_80GB",
            "Price_per_GPUh_EUR": 0.55,
            "Variable_Cost_per_GPUh_EUR": 0.12,
            "Support_Cost_pct_of_Revenue": 0.05,
            "Utilization_Multiplier": 1.10,
            "Mix_Share_of_Total_Demand": 0.30,
        },
        {
            "Product_ID": 3,
            "Product_Name": "Private GPU Cluster",
            "Product_Type": "DedicatedCluster",
            "GPU_SKU_Name": "NVIDIA_A100_80GB",
            "Price_per_GPUh_EUR": 0.40,
            "Variable_Cost_per_GPUh_EUR": 0.09,
            "Support_Cost_pct_of_Revenue": 0.02,
            "Utilization_Multiplier": 0.90,
            "Mix_Share_of_Total_Demand": 0.20,
        },
        {
            "Product_ID": 4,
            "Product_Name": "Managed AI Platform",
            "Product_Type": "ManagedService",
            "GPU_SKU_Name": "NVIDIA_H100_80GB",
            "Price_per_GPUh_EUR": 0.70,
            "Variable_Cost_per_GPUh_EUR": 0.15,
            "Support_Cost_pct_of_Revenue": 0.08,
            "Utilization_Multiplier": 0.85,
            "Mix_Share_of_Total_Demand": 0.10,
        },
    ]
    return pd.DataFrame(data)


def get_default_capacity_phases():
    data = [
        {
            "Phase_ID": 1,
            "Phase_Name": "Phase 1 - Initial Build",
            "Start_Year": 2025,
            "GPUs_Installed_in_Phase": 300,
            "Racks_in_Phase": 10,
        },
        {
            "Phase_ID": 2,
            "Phase_Name": "Phase 2 - Expansion",
            "Start_Year": 2027,
            "GPUs_Installed_in_Phase": 500,
            "Racks_in_Phase": 18,
        },
        {
            "Phase_ID": 3,
            "Phase_Name": "Phase 3 - Scale-Up",
            "Start_Year": 2029,
            "GPUs_Installed_in_Phase": 700,
            "Racks_in_Phase": 25,
        },
    ]
    return pd.DataFrame(data)


def get_default_investments():
    data = [
        {
            "Phase_ID": 1,
            "Phase_Name": "Phase 1 - Initial Build",
            "Start_Year": 2025,
            "CAPEX_IT_EUR": 3_000_000,
            "CAPEX_Building_EUR": 1_000_000,
            "CAPEX_Cooling_EUR": 500_000,
            "CAPEX_Network_EUR": 300_000,
        },
        {
            "Phase_ID": 2,
            "Phase_Name": "Phase 2 - Expansion",
            "Start_Year": 2027,
            "CAPEX_IT_EUR": 4_000_000,
            "CAPEX_Building_EUR": 800_000,
            "CAPEX_Cooling_EUR": 600_000,
            "CAPEX_Network_EUR": 400_000,
        },
        {
            "Phase_ID": 3,
            "Phase_Name": "Phase 3 - Scale-Up",
            "Start_Year": 2029,
            "CAPEX_IT_EUR": 5_000_000,
            "CAPEX_Building_EUR": 1_000_000,
            "CAPEX_Cooling_EUR": 800_000,
            "CAPEX_Network_EUR": 500_000,
        },
    ]
    df = pd.DataFrame(data)
    df["Total_CAPEX_EUR"] = (
        df["CAPEX_IT_EUR"]
        + df["CAPEX_Building_EUR"]
        + df["CAPEX_Cooling_EUR"]
        + df["CAPEX_Network_EUR"]
    )
    return df


def get_default_gpu_skus():
    data = [
        {
            "GPU_SKU_Name": "NVIDIA_A100_80GB",
            "GPU_Power_kW": 0.4,
            "Base_Variable_Cost_per_GPUh_EUR": 0.06,
            "Suggested_Price_per_GPUh_EUR": 0.45,
        },
        {
            "GPU_SKU_Name": "NVIDIA_H100_80GB",
            "GPU_Power_kW": 0.7,
            "Base_Variable_Cost_per_GPUh_EUR": 0.09,
            "Suggested_Price_per_GPUh_EUR": 0.70,
        },
        {
            "GPU_SKU_Name": "NVIDIA_L40S",
            "GPU_Power_kW": 0.35,
            "Base_Variable_Cost_per_GPUh_EUR": 0.05,
            "Suggested_Price_per_GPUh_EUR": 0.38,
        },
    ]
    return pd.DataFrame(data)


def get_default_tariffs():
    data = [
        {
            "Tariff_Name": "Base Tariff",
            "Start_Year": 2025,
            "Power_Price_EUR_per_kWh": 0.12,
        },
        {
            "Tariff_Name": "Higher Grid Tariff",
            "Start_Year": 2029,
            "Power_Price_EUR_per_kWh": 0.16,
        },
    ]
    return pd.DataFrame(data)


def get_scenario_presets():
    return {
        "Base Case": {
            "Base_SOM_Year1": 0.08,
            "SOM_Growth": 0.02,
            "Price_Multiplier": 1.0,
            "Utilization_Multiplier": 1.0,
            "Power_Price_Multiplier": 1.0,
        },
        "High Growth": {
            "Base_SOM_Year1": 0.10,
            "SOM_Growth": 0.03,
            "Price_Multiplier": 1.1,
            "Utilization_Multiplier": 1.05,
            "Power_Price_Multiplier": 1.0,
        },
        "Price Pressure": {
            "Base_SOM_Year1": 0.08,
            "SOM_Growth": 0.02,
            "Price_Multiplier": 0.85,
            "Utilization_Multiplier": 1.0,
            "Power_Price_Multiplier": 1.0,
        },
        "High Energy Cost": {
            "Base_SOM_Year1": 0.08,
            "SOM_Growth": 0.02,
            "Price_Multiplier": 1.0,
            "Utilization_Multiplier": 1.0,
            "Power_Price_Multiplier": 1.4,
        },
        "Enterprise Focus": {
            "Base_SOM_Year1": 0.12,
            "SOM_Growth": 0.025,
            "Price_Multiplier": 1.05,
            "Utilization_Multiplier": 0.95,
            "Power_Price_Multiplier": 1.0,
        },
    }


# -----------------------------
# Model Computation
# -----------------------------

def get_power_price_for_year(year, tariffs_df, scenario_power_mult, default_price):
    """Pick applicable tariff for given year, then apply scenario multiplier."""
    if tariffs_df is None or tariffs_df.empty:
        return default_price * scenario_power_mult
    valid = tariffs_df[tariffs_df["Start_Year"] <= year]
    if valid.empty:
        base = default_price
    else:
        base = valid.sort_values("Start_Year").iloc[-1]["Power_Price_EUR_per_kWh"]
    return base * scenario_power_mult


def compute_model(
    start_year: int,
    model_years: int,
    hours_per_year: float,
    discount_rate: float,
    base_TAM_gpus: float,
    base_SAM_share: float,
    base_SOM_year1: float,
    SOM_growth_per_year: float,
    base_gpu_utilization: float,
    base_power_price: float,
    global_gpu_power_kw: float,
    fixed_opex_per_year: float,
    other_variable_cost_per_gpu_h: float,
    tax_rate: float,
    capex_lifetime_years: int,
    products: pd.DataFrame,
    capacity_phases: pd.DataFrame,
    investments: pd.DataFrame,
    gpu_skus: pd.DataFrame,
    tariffs: pd.DataFrame,
    scenario: dict,
    scenario_name: str = "Scenario",
):
    years = np.arange(model_years)
    calendar_years = start_year + years

    # Scenario multipliers
    som_year1 = scenario["Base_SOM_Year1"]
    som_growth = scenario["SOM_Growth"]
    price_mult = scenario["Price_Multiplier"]
    util_mult = scenario["Utilization_Multiplier"]
    power_price_mult = scenario["Power_Price_Multiplier"]

    # Enrich products with GPU SKUs
    prod = products.copy()
    if "GPU_SKU_Name" not in prod.columns:
        prod["GPU_SKU_Name"] = ""
    skus = gpu_skus.copy()
    prod = prod.merge(skus, on="GPU_SKU_Name", how="left", suffixes=("", "_SKU"))

    # Fill defaults where SKU data missing
    prod["GPU_Power_kW_Effective"] = prod["GPU_Power_kW"].fillna(global_gpu_power_kw)
    prod["Variable_Cost_per_GPUh_EUR"] = prod["Variable_Cost_per_GPUh_EUR"].fillna(
        prod["Base_Variable_Cost_per_GPUh_EUR"]
    )

    prod["Price_per_GPUh_EUR"] *= price_mult
    prod["Effective_Utilization"] = (
        base_gpu_utilization * prod["Utilization_Multiplier"] * util_mult
    )

    # Market demand per year & product
    rows = []
    for i, year in enumerate(calendar_years):
        som_share = som_year1 + i * som_growth
        total_demand_gpus = base_TAM_gpus * base_SAM_share * som_share

        for _, p in prod.iterrows():
            gpu_demand_prod = total_demand_gpus * p["Mix_Share_of_Total_Demand"]
            gpu_hours_sold = gpu_demand_prod * p["Effective_Utilization"] * hours_per_year

            rows.append(
                {
                    "Scenario": scenario_name,
                    "Year_Index": i + 1,
                    "Calendar_Year": year,
                    "Product_ID": p["Product_ID"],
                    "Product_Name": p["Product_Name"],
                    "GPU_SKU_Name": p["GPU_SKU_Name"],
                    "SOM_Share": som_share,
                    "Total_Demand_GPUs": total_demand_gpus,
                    "Product_Mix_Share": p["Mix_Share_of_Total_Demand"],
                    "GPU_Demand_for_Product": gpu_demand_prod,
                    "Effective_Utilization": p["Effective_Utilization"],
                    "GPU_Hours_Sold": gpu_hours_sold,
                    "Price_per_GPUh_EUR": p["Price_per_GPUh_EUR"],
                    "Variable_Cost_per_GPUh_EUR": p["Variable_Cost_per_GPUh_EUR"],
                    "Support_Cost_pct_of_Revenue": p["Support_Cost_pct_of_Revenue"],
                    "GPU_Power_kW_Effective": p["GPU_Power_kW_Effective"],
                }
            )

    df_demand = pd.DataFrame(rows)

    # Capacity per year (still uses global GPU power for kW sizing)
    cap_ph = capacity_phases.copy()
    cap_rows = []
    for _, ph in cap_ph.iterrows():
        ph["kW_Load_in_Phase"] = ph["GPUs_Installed_in_Phase"] * global_gpu_power_kw
    for i, year in enumerate(calendar_years):
        active = cap_ph[cap_ph["Start_Year"] <= year]
        total_gpus = active["GPUs_Installed_in_Phase"].sum()
        total_kw = (active["GPUs_Installed_in_Phase"] * global_gpu_power_kw).sum()
        cap_rows.append(
            {
                "Scenario": scenario_name,
                "Year_Index": i + 1,
                "Calendar_Year": year,
                "Total_Capacity_GPUs": total_gpus,
                "Total_Capacity_kW": total_kw,
                "Total_GPU_Hours_Theoretical": total_gpus * hours_per_year,
                "Total_GPU_Hours_at_Base_Utilization": total_gpus * hours_per_year * base_gpu_utilization,
            }
        )
    df_capacity = pd.DataFrame(cap_rows)

    # Investment & depreciation
    inv = investments.copy()
    inv["Annual_Depreciation_EUR"] = inv["Total_CAPEX_EUR"] / capex_lifetime_years

    fin_rows = []
    for i, year in enumerate(calendar_years):
        year_index = i + 1
        df_y = df_demand[df_demand["Year_Index"] == year_index]

        total_gpu_hours = df_y["GPU_Hours_Sold"].sum()
        revenue = (df_y["GPU_Hours_Sold"] * df_y["Price_per_GPUh_EUR"]).sum()

        # Tariff-based power price
        year_power_price = get_power_price_for_year(
            year, tariffs, power_price_mult, base_power_price
        )

        # Energy cost per product: GPU_Hours * GPU_Power_kW_Effective * price
        energy_cost = (df_y["GPU_Hours_Sold"] * df_y["GPU_Power_kW_Effective"] * year_power_price).sum()

        other_var_costs = total_gpu_hours * other_variable_cost_per_gpu_h
        support_costs = (
            df_y["GPU_Hours_Sold"]
            * df_y["Price_per_GPUh_EUR"]
            * df_y["Support_Cost_pct_of_Revenue"]
        ).sum()
        product_specific_var = (
            df_y["GPU_Hours_Sold"] * df_y["Variable_Cost_per_GPUh_EUR"]
        ).sum()

        gross_margin = revenue - energy_cost - other_var_costs - support_costs - product_specific_var
        fixed_opex = fixed_opex_per_year
        EBITDA = gross_margin - fixed_opex

        dep_active = inv[inv["Start_Year"] <= year]["Annual_Depreciation_EUR"].sum()
        EBIT = EBITDA - dep_active

        tax = max(EBIT * tax_rate, 0)
        net_income = EBIT - tax

        capex_this_year = inv[inv["Start_Year"] == year]["Total_CAPEX_EUR"].sum()
        free_cash_flow = net_income + dep_active - capex_this_year

        fin_rows.append(
            {
                "Scenario": scenario_name,
                "Year_Index": year_index,
                "Calendar_Year": year,
                "Total_GPU_Hours_Sold": total_gpu_hours,
                "Revenue_EUR": revenue,
                "Energy_Cost_EUR": energy_cost,
                "Other_Variable_Costs_EUR": other_var_costs,
                "Product_Variable_Costs_EUR": product_specific_var,
                "Support_Costs_EUR": support_costs,
                "Gross_Margin_EUR": gross_margin,
                "Fixed_OPEX_EUR": fixed_opex,
                "EBITDA_EUR": EBITDA,
                "Depreciation_EUR": dep_active,
                "EBIT_EUR": EBIT,
                "Tax_EUR": tax,
                "Net_Income_EUR": net_income,
                "CAPEX_EUR": capex_this_year,
                "Free_Cash_Flow_EUR": free_cash_flow,
            }
        )

    df_fin = pd.DataFrame(fin_rows)

    cash_flows = df_fin["Free_Cash_Flow_EUR"].values
    discount_factors = 1 / (1 + discount_rate) ** np.arange(1, model_years + 1)
    npv = float((cash_flows * discount_factors).sum())

    irr = None
    try:
        irr = float(
            np.irr(
                np.concatenate(
                    ([-investments["Total_CAPEX_EUR"].sum()], cash_flows)
                )
            )
        )
    except Exception:
        irr = None

    summary = {
        "Scenario": scenario_name,
        "NPV_EUR": npv,
        "IRR": irr,
        "Peak_Revenue": float(df_fin["Revenue_EUR"].max()),
        "Peak_EBITDA": float(df_fin["EBITDA_EUR"].max()),
        "Last_Year_Revenue": float(df_fin["Revenue_EUR"].iloc[-1]),
        "Last_Year_EBITDA": float(df_fin["EBITDA_EUR"].iloc[-1]),
    }

    return df_demand, df_capacity, df_fin, summary


# # -----------------------------
# # Excel Import/Export
# # -----------------------------

# def load_from_excel(file) -> dict:
#     """Expect sheets: Global_Inputs, Products, Capacity, Investment."""
#     xls = pd.ExcelFile(file)
#     out = {}

#     if "Global_Inputs" in xls.sheet_names:
#         gi = pd.read_excel(xls, "Global_Inputs")
#         gi = gi.set_index("Parameter")["Value"]

#         def g(param, default=None):
#             return gi.get(param, default)

#         out["start_year"] = int(g("Model_Start_Year", 2025))
#         out["model_years"] = int(g("Model_Years", 10))
#         out["base_TAM_gpus"] = float(g("Base_TAM_GPUs", 5460))
#         out["base_SAM_share"] = float(g("Base_SAM_Share", 0.85))
#         out["base_gpu_utilization"] = float(g("Base_GPU_Utilization", 0.65))
#         out["base_power_price"] = float(g("Base_Power_Price_per_kWh", 0.12))
#         out["global_gpu_power_kw"] = float(g("GPU_Power_kW", 0.7))
#         out["fixed_opex_per_year"] = float(g("Fixed_OPEX_per_Year", 500000))
#         out["other_var_cost"] = float(g("Other_Variable_Cost_per_GPUh", 0.02))
#         out["discount_rate"] = float(g("Discount_Rate", 0.10))
#         out["tax_rate"] = float(g("Corporate_Tax_Rate", 0.25))
#         out["capex_lifetime_years"] = int(g("CAPEX_Lifetime_Years", 5))

#     if "Products" in xls.sheet_names:
#         out["products_df"] = pd.read_excel(xls, "Products")

#     if "Capacity" in xls.sheet_names:
#         out["capacity_df"] = pd.read_excel(xls, "Capacity")

#     if "Investment" in xls.sheet_names:
#         inv_df = pd.read_excel(xls, "Investment")
#         if "Total_CAPEX_EUR" not in inv_df.columns:
#             inv_df["Total_CAPEX_EUR"] = (
#                 inv_df["CAPEX_IT_EUR"]
#                 + inv_df["CAPEX_Building_EUR"]
#                 + inv_df["CAPEX_Cooling_EUR"]
#                 + inv_df["CAPEX_Network_EUR"]
#             )
#         out["invest_df"] = inv_df

#     return out


# def export_to_excel(
#     global_config: dict,
#     products: pd.DataFrame,
#     capacity: pd.DataFrame,
#     investments: pd.DataFrame,
#     demand: pd.DataFrame,
#     capacity_summary: pd.DataFrame,
#     financials: pd.DataFrame,
# ) -> bytes:
#     buf = io.BytesIO()
#     with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
#         # Global_Inputs
#         gi_rows = [
#             ("Model_Start_Year", global_config["start_year"], "Year"),
#             ("Model_Years", global_config["model_years"], "Years"),
#             ("Hours_per_Year", 8760, "Hours"),
#             ("Discount_Rate", global_config["discount_rate"], ""),
#             ("Base_TAM_GPUs", global_config["base_TAM_gpus"], "GPUs"),
#             ("Base_SAM_Share", global_config["base_SAM_share"], ""),
#             ("Base_GPU_Utilization", global_config["base_gpu_utilization"], ""),
#             ("Base_Power_Price_per_kWh", global_config["base_power_price"], "EUR/kWh"),
#             ("GPU_Power_kW", global_config["global_gpu_power_kw"], "kW"),
#             ("Fixed_OPEX_per_Year", global_config["fixed_opex_per_year"], "EUR"),
#             ("Other_Variable_Cost_per_GPUh", global_config["other_var_cost"], "EUR/GPUh"),
#             ("Corporate_Tax_Rate", global_config["tax_rate"], ""),
#             ("CAPEX_Lifetime_Years", global_config["capex_lifetime_years"], "Years"),
#         ]
#         gi_df = pd.DataFrame(gi_rows, columns=["Parameter", "Value", "Unit"])
#         gi_df.to_excel(writer, sheet_name="Global_Inputs", index=False)

#         products.to_excel(writer, sheet_name="Products", index=False)
#         capacity.to_excel(writer, sheet_name="Capacity", index=False)
#         investments.to_excel(writer, sheet_name="Investment", index=False)
#         demand.to_excel(writer, sheet_name="Market_Demand", index=False)
#         capacity_summary.to_excel(writer, sheet_name="Capacity_Summary", index=False)
#         financials.to_excel(writer, sheet_name="Financials", index=False)

#     buf.seek(0)
#     return buf.getvalue()


# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.set_page_config(page_title="AI Factory Digital Twin", layout="wide")
    st.title("🤖 V53 Digital Twin – Scenario Simulator")

    if "products_df" not in st.session_state:
        st.session_state["products_df"] = get_default_products()
    if "capacity_df" not in st.session_state:
        st.session_state["capacity_df"] = get_default_capacity_phases()
    if "invest_df" not in st.session_state:
        st.session_state["invest_df"] = get_default_investments()
    if "gpu_skus_df" not in st.session_state:
        st.session_state["gpu_skus_df"] = get_default_gpu_skus()
    if "tariffs_df" not in st.session_state:
        st.session_state["tariffs_df"] = get_default_tariffs()

    # Sidebar: Global Inputs & Scenario
    st.sidebar.header("Global Assumptions")

    start_year = st.sidebar.number_input("Start Year", value=2025, step=1)
    model_years = st.sidebar.slider("Model Years", min_value=3, max_value=20, value=10)

    base_TAM_gpus = st.sidebar.number_input("Base TAM (GPUs)", value=5460.0, step=100.0)
    base_SAM_share = st.sidebar.slider("SAM Share (0–1)", min_value=0.0, max_value=1.0, value=0.85)

    base_gpu_utilization = st.sidebar.slider("Base GPU Utilization", 0.1, 0.99, 0.65)
    base_power_price = st.sidebar.number_input("Base Power Price (EUR/kWh)", value=0.12, step=0.01)
    global_gpu_power_kw = st.sidebar.number_input("Global GPU Power (kW per GPU)", value=0.7, step=0.1)

    fixed_opex_per_year = st.sidebar.number_input("Fixed OPEX per Year (EUR)", value=500_000.0, step=50_000.0)
    other_variable_cost_per_gpu_h = st.sidebar.number_input(
        "Other Var. Cost per GPUh (EUR)", value=0.02, step=0.01
    )

    discount_rate = st.sidebar.slider("Discount Rate (WACC)", 0.02, 0.30, 0.10)
    tax_rate = st.sidebar.slider("Tax Rate", 0.0, 0.50, 0.25)
    capex_lifetime_years = st.sidebar.number_input(
        "CAPEX Lifetime (years)", value=5, min_value=1, max_value=15, step=1
    )

    hours_per_year = 8760

    # Scenario selection (multi-select for comparison)
    st.sidebar.header("Scenarios")
    scenarios = get_scenario_presets()
    selected_scenarios = st.sidebar.multiselect(
        "Scenario Presets",
        list(scenarios.keys()),
        default=["Base Case", "High Growth"],
    )
    if not selected_scenarios:
        selected_scenarios = ["Base Case"]

    # File upload: load from Excel
    # st.sidebar.header("Import from Excel")
    # uploaded = st.sidebar.file_uploader("Upload scenario Excel", type=["xlsx"])
    # if uploaded is not None:
    #     try:
    #         loaded = load_from_excel(uploaded)
    #         if "start_year" in loaded:
    #             start_year = loaded["start_year"]
    #         if "model_years" in loaded:
    #             model_years = loaded["model_years"]
    #         if "base_TAM_gpus" in loaded:
    #             base_TAM_gpus = loaded["base_TAM_gpus"]
    #         if "base_SAM_share" in loaded:
    #             base_SAM_share = loaded["base_SAM_share"]
    #         if "base_gpu_utilization" in loaded:
    #             base_gpu_utilization = loaded["base_gpu_utilization"]
    #         if "base_power_price" in loaded:
    #             base_power_price = loaded["base_power_price"]
    #         if "global_gpu_power_kw" in loaded:
    #             global_gpu_power_kw = loaded["global_gpu_power_kw"]
    #         if "fixed_opex_per_year" in loaded:
    #             fixed_opex_per_year = loaded["fixed_opex_per_year"]
    #         if "other_var_cost" in loaded:
    #             other_variable_cost_per_gpu_h = loaded["other_var_cost"]
    #         if "discount_rate" in loaded:
    #             discount_rate = loaded["discount_rate"]
    #         if "tax_rate" in loaded:
    #             tax_rate = loaded["tax_rate"]
    #         if "capex_lifetime_years" in loaded:
    #             capex_lifetime_years = loaded["capex_lifetime_years"]
    #         if "products_df" in loaded:
    #             st.session_state["products_df"] = loaded["products_df"]
    #         if "capacity_df" in loaded:
    #             st.session_state["capacity_df"] = loaded["capacity_df"]
    #         if "invest_df" in loaded:
    #             st.session_state["invest_df"] = loaded["invest_df"]
    #         st.sidebar.success("Excel model imported successfully.")
    #     except Exception as e:
    #         st.sidebar.error(f"Failed to import: {e}")

    # Configuration section
    st.subheader("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Products (incl. GPU SKUs)")
        st.session_state["products_df"] = st.data_editor(
            st.session_state["products_df"],
            use_container_width=True,
            num_rows="dynamic",
            key="products_editor",
        )

        st.markdown("### GPU SKUs (real hardware)")
        st.session_state["gpu_skus_df"] = st.data_editor(
            st.session_state["gpu_skus_df"],
            use_container_width=True,
            num_rows="dynamic",
            key="gpus_editor",
        )

    with col2:
        st.markdown("### Capacity Phases")
        st.session_state["capacity_df"] = st.data_editor(
            st.session_state["capacity_df"],
            use_container_width=True,
            num_rows="dynamic",
            key="capacity_editor",
        )

        st.markdown("### Power Tariffs (real-world grid contracts)")
        st.session_state["tariffs_df"] = st.data_editor(
            st.session_state["tariffs_df"],
            use_container_width=True,
            num_rows="dynamic",
            key="tariffs_editor",
        )

    st.markdown("### Investment Phases")
    st.session_state["invest_df"] = st.data_editor(
        st.session_state["invest_df"],
        use_container_width=True,
        num_rows="dynamic",
        key="investment_editor",
    )
    invest_df = st.session_state["invest_df"].copy()
    if {"CAPEX_IT_EUR", "CAPEX_Building_EUR", "CAPEX_Cooling_EUR", "CAPEX_Network_EUR"}.issubset(
        invest_df.columns
    ):
        invest_df["Total_CAPEX_EUR"] = (
            invest_df["CAPEX_IT_EUR"]
            + invest_df["CAPEX_Building_EUR"]
            + invest_df["CAPEX_Cooling_EUR"]
            + invest_df["CAPEX_Network_EUR"]
        )

    # Run model for each scenario selected
    all_demand = []
    all_capacity = []
    all_financials = []
    summaries = []

    for scen_name in selected_scenarios:
        scen = scenarios[scen_name]
        demand_df, capacity_df, fin_df, summary = compute_model(
            start_year=start_year,
            model_years=model_years,
            hours_per_year=hours_per_year,
            discount_rate=discount_rate,
            base_TAM_gpus=base_TAM_gpus,
            base_SAM_share=base_SAM_share,
            base_SOM_year1=scen["Base_SOM_Year1"],
            SOM_growth_per_year=scen["SOM_Growth"],
            base_gpu_utilization=base_gpu_utilization,
            base_power_price=base_power_price,
            global_gpu_power_kw=global_gpu_power_kw,
            fixed_opex_per_year=fixed_opex_per_year,
            other_variable_cost_per_gpu_h=other_variable_cost_per_gpu_h,
            tax_rate=tax_rate,
            capex_lifetime_years=capex_lifetime_years,
            products=st.session_state["products_df"],
            capacity_phases=st.session_state["capacity_df"],
            investments=invest_df,
            gpu_skus=st.session_state["gpu_skus_df"],
            tariffs=st.session_state["tariffs_df"],
            scenario=scen,
            scenario_name=scen_name,
        )
        all_demand.append(demand_df)
        all_capacity.append(capacity_df)
        all_financials.append(fin_df)
        summaries.append(summary)

    demand_all = pd.concat(all_demand, ignore_index=True)
    capacity_all = pd.concat(all_capacity, ignore_index=True)
    fin_all = pd.concat(all_financials, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    # KPIs
    st.subheader("Key Financial KPIs (per scenario)")
    st.dataframe(summary_df, use_container_width=True)

    # Export button
    global_config = dict(
        start_year=start_year,
        model_years=model_years,
        base_TAM_gpus=base_TAM_gpus,
        base_SAM_share=base_SAM_share,
        base_gpu_utilization=base_gpu_utilization,
        base_power_price=base_power_price,
        global_gpu_power_kw=global_gpu_power_kw,
        fixed_opex_per_year=fixed_opex_per_year,
        other_var_cost=other_variable_cost_per_gpu_h,
        discount_rate=discount_rate,
        tax_rate=tax_rate,
        capex_lifetime_years=capex_lifetime_years,
    )

    # export_bytes = export_to_excel(
    #     global_config=global_config,
    #     products=st.session_state["products_df"],
    #     capacity=st.session_state["capacity_df"],
    #     investments=invest_df,
    #     demand=demand_all,
    #     capacity_summary=capacity_all,
    #     financials=fin_all,
    # )

    # st.download_button(
    #     "📥 Download current model as Excel",
    #     data=export_bytes,
    #     file_name="AI_Factory_Digital_Twin.xlsx",
    #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # )

    # # Tabs for detailed views
    # tab_overview, tab_demand, tab_capacity, tab_financials = st.tabs(
    #     ["Overview (Multi-Scenario)", "Market & Demand", "Capacity", "Financials"]
    # )

    with tab_overview:
        st.markdown("### Revenue & EBITDA over Time – Scenario Comparison")
        rev_pivot = fin_all.pivot_table(
            index="Calendar_Year", columns="Scenario", values="Revenue_EUR", aggfunc="sum"
        )
        ebitda_pivot = fin_all.pivot_table(
            index="Calendar_Year", columns="Scenario", values="EBITDA_EUR", aggfunc="sum"
        )
        st.line_chart(rev_pivot)
        st.line_chart(ebitda_pivot)

        st.markdown("### Free Cash Flow – Scenario Comparison")
        fcf_pivot = fin_all.pivot_table(
            index="Calendar_Year", columns="Scenario", values="Free_Cash_Flow_EUR", aggfunc="sum"
        )
        st.bar_chart(fcf_pivot)

    with tab_demand:
        st.markdown("### Market Demand by Year, Product and Scenario")
        st.dataframe(demand_all, use_container_width=True, height=400)

        st.markdown("### GPU-Hours Sold by Product (Base Case only for clarity)")
        base_only = demand_all[demand_all["Scenario"] == selected_scenarios[0]]
        pivot = (
            base_only.pivot_table(
                index="Calendar_Year",
                columns="Product_Name",
                values="GPU_Hours_Sold",
                aggfunc="sum",
            )
            .fillna(0)
        )
        st.area_chart(pivot)

    with tab_capacity:
        st.markdown("### Capacity by Year (Base Case)")
        base_cap = capacity_all[capacity_all["Scenario"] == selected_scenarios[0]]
        st.dataframe(base_cap, use_container_width=True, height=300)

        st.markdown("### Total Capacity (GPUs) – Base Case")
        cap_plot = base_cap[["Calendar_Year", "Total_Capacity_GPUs"]].set_index("Calendar_Year")
        st.bar_chart(cap_plot)

        st.markdown("### Capacity vs Demand – Approx GPUs (Base Case)")
        base_fin = fin_all[fin_all["Scenario"] == selected_scenarios[0]].copy()
        approx_demand_gpus = base_fin["Total_GPU_Hours_Sold"] / (
            hours_per_year * base_gpu_utilization
        )
        capd = base_cap.copy()
        capd["Approx_Demand_GPUs"] = approx_demand_gpus.values
        comp = capd[
            ["Calendar_Year", "Total_Capacity_GPUs", "Approx_Demand_GPUs"]
        ].set_index("Calendar_Year")
        st.line_chart(comp)

    with tab_financials:
        st.markdown("### Financial Statements by Year and Scenario")
        st.dataframe(fin_all, use_container_width=True, height=400)

        st.markdown("### Free Cash Flow (Base Case)")
        base_fin = fin_all[fin_all["Scenario"] == selected_scenarios[0]]
        fcf_plot = base_fin[["Calendar_Year", "Free_Cash_Flow_EUR"]].set_index("Calendar_Year")
        st.bar_chart(fcf_plot)


if __name__ == "__main__":
    main()
