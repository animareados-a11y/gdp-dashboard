"""MarketSentinel display grid. Does not mutate scientific datasets."""
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

BUY = ('BUY1', 'BUY2', 'BUY3', 'BUY1(PR)', 'BUY2(PR)', 'BUY3(PR)')


def show_grid(df, key, portfolio_mode=False):
    view = df.copy()
    view = view.loc[:, ~view.columns.duplicated()].reset_index(drop=True)
    daily = 'PHASE WEEKLY' in view.columns
    if 'MERCATO' in view:
        view['MERCATO'] = view['MERCATO'].astype('string').str.replace(r'^USA.*$', 'USA', regex=True)
    if daily:
        rename = {}
        if 'PHASE DAILY' not in view.columns:
            rename['PHASE NUOVA'] = 'PHASE DAILY'
        if 'RATING DAILY' not in view.columns:
            rename['RATING'] = 'RATING DAILY'
        view = view.rename(columns=rename)
        wanted = ['NOME TITOLO','PREZZO','MERCATO','PHASE DAILY','RATING DAILY','PHASE WEEKLY','RATING WEEKLY','RENDIMENTO ATTESO','DATA']
    else:
        view = view.rename(columns={'PHASE NUOVA': 'PHASE WEEKLY', 'RATING': 'RATING WEEKLY'})
        wanted = ['NOME TITOLO','PREZZO','MERCATO','PHASE WEEKLY','RATING WEEKLY','RENDIMENTO ATTESO','DATA']
    view = view.loc[:, ~view.columns.duplicated()]
    view = view[[c for c in wanted if c in view.columns]]
    for c in ('PREZZO','RATING DAILY','RATING WEEKLY','RENDIMENTO ATTESO'):
        if c in view:
            view[c] = pd.to_numeric(view[c], errors='coerce')
    if 'DATA' in view:
        dt = pd.to_datetime(view['DATA'], dayfirst=True, errors='coerce')
        view['DATA'] = dt.dt.strftime('%d/%m/%Y').fillna(view['DATA'].astype(str))
    gb = GridOptionsBuilder.from_dataframe(view)
    gb.configure_default_column(sortable=True, resizable=True, filter=True, floatingFilter=False, minWidth=110)
    gb.configure_grid_options(animateRows=True, enableCellTextSelection=True, suppressRowClickSelection=True)
    gb.configure_pagination(enabled=False)
    gb.configure_grid_options(domLayout='normal', suppressHorizontalScroll=False,
                              alwaysShowHorizontalScroll=True, alwaysShowVerticalScroll=True)
    if portfolio_mode and daily:
        view.insert(0, 'P', 'P')
        gb = GridOptionsBuilder.from_dataframe(view)
        gb.configure_default_column(sortable=True, resizable=True, filter=True, minWidth=110)
        gb.configure_grid_options(
            animateRows=True,
            enableCellTextSelection=True,
            suppressRowClickSelection=True,
            rowSelection='single',
            pagination=False,
            domLayout='normal',
            suppressHorizontalScroll=False,
            alwaysShowHorizontalScroll=True,
            alwaysShowVerticalScroll=True,
        )
        gb.configure_column(
            'P', header_name='P', pinned='left',
            width=55, minWidth=55, maxWidth=55,
            sortable=False, filter=False,
            checkboxSelection=True,
            headerCheckboxSelection=False,
        )
    gb.configure_column('NOME TITOLO', header_name='TITOLO', pinned='left', minWidth=145)
    if "PREZZO" in view:
        gb.configure_column(
            "PREZZO", type=["numericColumn"],
            filter="agNumberColumnFilter", minWidth=100,
            valueFormatter=JsCode(
                "function(p){if(p.value == null)return 'N/D';"
                "const m=String(p.data.MERCATO||'').toUpperCase();"
                "return (m.startsWith('USA')?'$ ':'€ ')+Number(p.value).toFixed(2);}"
            )
        )
    # AG Grid Set Filter is an Enterprise feature. Enable enterprise modules to show
    # the Excel-style checkbox menu; production use may require an AG Grid licence.
    for c in ('MERCATO','PHASE DAILY','PHASE WEEKLY','DATA'):
        if c in view:
            gb.configure_column(c, filter='agSetColumnFilter', filterParams={'buttons':['reset','apply'], 'closeOnApply':True})
    # MS_DAILY_BULLISH_CONFLUENCE_YELLOW_V1
    # Evidenzia solo le righe DAILY in cui entrambe le PHASE sono rialziste.
    bullish_js = """function(p) {
        if (!p || !p.data) return false;
        const allowed = ['BUY1','BUY2','BUY3','BUY1(PR)','BUY2(PR)','BUY3(PR)','TREND_RIALZISTA','DEBOLEZZA_RIALZISTA'];
        const norm = (x) => String(x == null ? '' : x).trim().toUpperCase()
            .replace(/\s+/g, '_').replace(/^(BUY|SELL)_(\d)$/, '$1$2');
        return allowed.includes(norm(p.data['PHASE DAILY'])) &&
               allowed.includes(norm(p.data['PHASE WEEKLY']));
    }"""
    if daily and 'PHASE DAILY' in view and 'PHASE WEEKLY' in view:
        gb.configure_grid_options(getRowStyle=JsCode("""function(p) {
            const allowed = ['BUY1','BUY2','BUY3','BUY1(PR)','BUY2(PR)','BUY3(PR)','TREND_RIALZISTA','DEBOLEZZA_RIALZISTA'];
            const norm = (x) => String(x == null ? '' : x).trim().toUpperCase()
                .replace(/\s+/g, '_').replace(/^(BUY|SELL)_(\d)$/, '$1$2');
            return p.data && allowed.includes(norm(p.data['PHASE DAILY'])) &&
                allowed.includes(norm(p.data['PHASE WEEKLY'])) ?
                {'backgroundColor':'#fff2a8','color':'#242424'} : undefined;
        }"""))
    buy_style = JsCode("""function(p) {
        const allowed = ['BUY1','BUY2','BUY3','BUY1(PR)','BUY2(PR)','BUY3(PR)','TREND_RIALZISTA','DEBOLEZZA_RIALZISTA'];
        const norm = (x) => String(x == null ? '' : x).trim().toUpperCase()
            .replace(/\s+/g, '_').replace(/^(BUY|SELL)_(\d)$/, '$1$2');
        if (p.data && p.data['PHASE DAILY'] != null && p.data['PHASE WEEKLY'] != null &&
            allowed.includes(norm(p.data['PHASE DAILY'])) &&
            allowed.includes(norm(p.data['PHASE WEEKLY']))) {
            return {'backgroundColor':'#fff2a8','color':'#242424','fontWeight':'bold'};
        }
        return ['BUY1','BUY2','BUY3','BUY1(PR)','BUY2(PR)','BUY3(PR)'].includes(p.value) ?
            {'backgroundColor':'#d9f5e9','color':'#075c43','fontWeight':'bold'} :
            (String(p.value||'').startsWith('SELL') ? {'backgroundColor':'#fde7e8','color':'#922f38'} : {});
    }""")
    for c in ('PHASE DAILY','PHASE WEEKLY'):
        if c in view:
            gb.configure_column(c, cellStyle=buy_style, minWidth=175)
    for c in ('RATING DAILY','RATING WEEKLY'):
        if c in view:
            gb.configure_column(c, type=['numericColumn'], filter='agNumberColumnFilter', valueFormatter=JsCode("function(p){return p.value == null || p.value === '' ? '—' : Number(p.value).toFixed(3);}"), minWidth=135)
    if 'RENDIMENTO ATTESO' in view:
        gb.configure_column('RENDIMENTO ATTESO', type=['numericColumn'], filter='agNumberColumnFilter', minWidth=180,
            valueFormatter=JsCode("function(p){return p.value == null || p.value === '' ? 'N/D' : (Number(p.value)>=0?'+':'')+Number(p.value).toFixed(2)+'%';}"),
            cellStyle=JsCode("function(p){return p.value == null ? {} : {'color':Number(p.value)>=0?'#067354':'#b33948','fontWeight':'bold'};}"))
    # Explicit scrollbar sizing, including the AG Grid horizontal scroll track.
    scrollbar_css = {
        '.ag-body-horizontal-scroll': {'min-height': '20px !important', 'height': '20px !important'},
        '.ag-body-horizontal-scroll-viewport': {'min-height': '20px !important', 'height': '20px !important'},
        '.ag-body-vertical-scroll': {'min-width': '20px !important', 'width': '20px !important'},
        '.ag-body-vertical-scroll-viewport': {'min-width': '20px !important', 'width': '20px !important'},
        '.ag-root-wrapper ::-webkit-scrollbar': {'width': '18px', 'height': '18px'},
        '.ag-root-wrapper ::-webkit-scrollbar-thumb': {
            'background-color': '#64869a', 'border-radius': '8px', 'border': '3px solid #dce8ee'
        },
        '.ag-root-wrapper ::-webkit-scrollbar-track': {'background-color': '#dce8ee'},
    }
    return AgGrid(view, gridOptions=gb.build(), theme='alpine', height=660, key=key,
                  update_on=['selectionChanged'] if portfolio_mode and daily else [],
                  custom_css=scrollbar_css,
                  allow_unsafe_jscode=True, enable_enterprise_modules=True,
                  fit_columns_on_grid_load=False)
