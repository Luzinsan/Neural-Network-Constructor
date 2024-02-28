import dearpygui.dearpygui as dpg
dpg.create_context()
# callback runs when user attempts to connect attributes
def link_callback(sender, app_data):
    # app_data -> (link_id1, link_id2)
    dpg.add_node_link(app_data[0], app_data[1], parent=sender)

# callback runs when user attempts to disconnect attributes
def delink_callback(sender, app_data):
    # app_data -> link_id
    dpg.delete_item(app_data)

# with dpg.window(label="Tutorial", width=400, height=400):
#     with dpg.node_editor(callback=link_callback, delink_callback=delink_callback):
#         with dpg.node(label="Node 1"):
#             with dpg.node_attribute(label="Node A1"):
#                 dpg.add_input_float(label="F1", width=150)
#             with dpg.node_attribute(label="Node A2", attribute_type=dpg.mvNode_Attr_Output):
#                 dpg.add_input_float(label="F2", width=150)
#         with dpg.node(label="Node 2"):
#             with dpg.node_attribute(label="Node A3"):
#                 dpg.add_input_float(label="F3", width=200)
#             with dpg.node_attribute(label="Node A4", attribute_type=dpg.mvNode_Attr_Output):
#                 dpg.add_input_float(label="F4", width=200)
    


# with dpg.window(label="Tab Bar", width=400, height=400):
#     with dpg.tab_bar():
#         with dpg.tab(label='lllabel'):
#             dpg.add_button(label="Great")
#             with dpg.tree_node(label='Tree node'):
#                 dpg.add_button(label='Tree Node 1')
#         with dpg.tab(label='lllabel2'):
#             dpg.add_button(tag="Great2")

#             with dpg.tooltip(parent='Great2'):
#                 dpg.add_text("A tooltip")
            
#         dpg.add_tab_button(label="Tab-button")
    

def _hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h * 6.)  # XXX assume int() truncates!
    f = (h * 6.) - i
    p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
    i %= 6
    if i == 0: return (255 * v, 255 * t, 255 * p)
    if i == 1: return (255 * q, 255 * v, 255 * p)
    if i == 2: return (255 * p, 255 * v, 255 * t)
    if i == 3: return (255 * p, 255 * q, 255 * v)
    if i == 4: return (255 * t, 255 * p, 255 * v)
    if i == 5: return (255 * v, 255 * p, 255 * q)


button_themes = 7 * [0]


def rlines():
    with dpg.child(width=200, height=90):
        item = dpg.add_button(label='RLINES')
        dpg.set_item_theme(dpg.last_item(), button_themes[5])
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text('Linhas de Resistencia')
        s = dpg.add_slider_float(label='%',
                                 default_value=2,
                                 min_value=0.1,
                                 max_value=5)
        i = dpg.add_input_int(default_value=0, label='')
    with dpg.drag_payload(parent=item,
                          drag_data=('RLINES', s, i),
                          payload_type='candle_plot'):
        dpg.add_text('RLINES')


def rsi():
    with dpg.child(width=200, height=90):
        item = dpg.add_button(label='RSI')
        dpg.set_item_theme(dpg.last_item(), button_themes[5])
        i = dpg.add_slider_int(default_value=14,
                               label='',
                               min_value=3,
                               max_value=200)
        j = dpg.add_slider_float(default_value=24,
                                 label='',
                                 min_value=3,
                                 max_value=200)
    with dpg.drag_payload(parent=item,
                          drag_data=('RSI', i, j),
                          payload_type='volume_plot'):
        dpg.add_text('RSI')


def sma():
    with dpg.child(width=200, height=70):
        item = dpg.add_button(label='SMA')
        dpg.set_item_theme(dpg.last_item(), button_themes[5])
        dpg.add_same_line()
        i = dpg.add_input_int(default_value=3, label='', width=70)
        dpg.add_button(label='SMA200')
        dpg.set_item_theme(dpg.last_item(), button_themes[4])
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text('Media Movel de 200 dias')
        with dpg.drag_payload(parent=dpg.last_item(),
                              drag_data=('SMA', 200),
                              payload_type='candle_plot'):
            dpg.add_text('SMA200')
        dpg.add_same_line()
        dpg.add_button(label='SMA50')
        dpg.set_item_theme(dpg.last_item(), button_themes[3])
        with dpg.drag_payload(parent=dpg.last_item(),
                              drag_data=('SMA', 50),
                              payload_type='candle_plot'):
            dpg.add_text('SMA50')
        dpg.add_same_line()
        dpg.add_button(label='SMA7')
        dpg.set_item_theme(dpg.last_item(), button_themes[2])
        with dpg.drag_payload(parent=dpg.last_item(),
                              drag_data=('SMA', 7),
                              payload_type='candle_plot'):
            dpg.add_text('SMA7')
    with dpg.drag_payload(parent=item,
                          drag_data=('SMA', dpg.get_value(i)),
                          payload_type='candle_plot'):
        dpg.add_text('SMA')


def ema():
    with dpg.child(width=200, height=70):
        item = dpg.add_button(label='EMA')
        dpg.set_item_theme(dpg.last_item(), button_themes[5])
        dpg.add_same_line()
        i = dpg.add_input_int(default_value=0, label='', width=70)
        dpg.add_button(label='EMA200')
        dpg.set_item_theme(dpg.last_item(), button_themes[4])
        with dpg.drag_payload(parent=dpg.last_item(),
                              drag_data=('EMA', 200),
                              payload_type='candle_plot'):
            dpg.add_text('EMA200')
        dpg.add_same_line()
        dpg.add_button(label='EMA50')
        dpg.set_item_theme(dpg.last_item(), button_themes[3])
        with dpg.drag_payload(parent=dpg.last_item(),
                              drag_data=('EMA', 50),
                              payload_type='candle_plot'):
            dpg.add_text('EMA50')
        dpg.add_same_line()
        dpg.add_button(label='EMA7')
        dpg.set_item_theme(dpg.last_item(), button_themes[2])
        with dpg.drag_payload(parent=dpg.last_item(),
                              drag_data=('EMA', 7),
                              payload_type='candle_plot'):
            dpg.add_text('EMA7')
    with dpg.drag_payload(parent=item,
                          drag_data=('EMA', dpg.get_value(i)),
                          payload_type='candle_plot'):
        dpg.add_text('EMA')


def bbands():
    with dpg.child(width=200, height=95):
        item = dpg.add_button(label='BOLLINGER BANDS')
        dpg.set_item_theme(dpg.last_item(), button_themes[5])
        i = dpg.add_slider_int(default_value=10,
                               label='',
                               min_value=3,
                               max_value=200)
        j = dpg.add_slider_float(default_value=5,
                                 label='',
                                 min_value=1,
                                 max_value=5)
    with dpg.drag_payload(parent=item,
                          drag_data=('BOLLINGER_BANDS', dpg.get_value(i),
                                     dpg.get_value(j)),
                          payload_type='candle_plot'):
        dpg.add_text('BOLLINGER BANDS')


def add_sma_series(n, dates, parent, ta):
    sma_dates = dates[n - 1:]
    sma_values = ta.sma(n).to_list()[n - 1:]
    dpg.add_line_series(sma_dates,
                        sma_values,
                        label='SMA-' + str(n),
                        parent=parent)
    dpg.add_button(label="Delete Series",
                   user_data=dpg.last_item(),
                   parent=dpg.last_item(),
                   callback=lambda s, a, u: dpg.delete_item(u))


def add_ema_series(n, dates, parent, ta):
    ema_dates = dates[n - 1:]
    ema_values = ta.ema(n).to_list()[n - 1:]
    dpg.add_line_series(ema_dates,
                        ema_values,
                        label='EMA-' + str(n),
                        parent=parent)
    dpg.add_button(label="Delete Series",
                   user_data=dpg.last_item(),
                   parent=dpg.last_item(),
                   callback=lambda s, a, u: dpg.delete_item(u))


def add_rsi_series(iw, ws, dates, ta, parent):
    values = ta.rsi(iw, ws)
    dpg.add_line_series(dates, values, label='RSI', parent=parent)
    dpg.add_button(label="Delete Series",
                   user_data=dpg.last_item(),
                   parent=dpg.last_item(),
                   callback=lambda s, a, u: dpg.delete_item(u))


def add_bbands_series(w, s, dates, ta, parent):
    values = ta.bollinger_bands(w, s)
    bbdates = dates[w - 1:]
    upper_band = values[1].to_list()[w - 1:]
    lower_band = values[2].to_list()[w - 1:]
    with dpg.theme() as b_theme:
        dpg.add_theme_color(dpg.mvPlotCol_Fill, (100, 100, 150, 64),
                            category=dpg.mvThemeCat_Plots)
    dpg.add_shade_series(bbdates,
                         upper_band,
                         y2=lower_band,
                         label="BBANDS",
                         parent=parent)
    dpg.set_item_theme(dpg.last_item(), b_theme)
    dpg.add_button(label="Delete Series",
                   user_data=dpg.last_item(),
                   parent=dpg.last_item(),
                   callback=lambda s, a, u: dpg.delete_item(u))
    
sma()


############### Developer Tools ##############
# dpg.show_documentation()
# dpg.show_style_editor()
# dpg.show_debug()
# dpg.show_about()
# dpg.show_metrics()
# dpg.show_font_manager()
# dpg.show_item_registry()
##############################################


dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()