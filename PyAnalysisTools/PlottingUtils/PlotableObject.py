import ROOT

color_palette = [
    # ROOT.kGray+3,
    # ROOT.kPink+7,
    # ROOT.kAzure+4,
    # ROOT.kSpring-9,
    # ROOT.kOrange-3,
    # ROOT.kCyan-6,
    # ROOT.kPink-7,
    # ROOT.kSpring-7,
    # ROOT.kPink-1

    # ROOT.kBlue+3,
    # ROOT.kBlue+3,
    
    ROOT.kGray+3,
    ROOT.kRed+2,
    ROOT.kAzure+4,
    ROOT.kSpring-6,
    ROOT.kOrange-3,
    ROOT.kCyan-3,
    ROOT.kPink-2,
    ROOT.kSpring-9,
    ROOT.kMagenta-5
]

marker_style_palette_filled = [#21,
                               20,
                               22,
                               23,
                               33,
                               34,
                               29,
                               2
]
marker_style_palette_empty = [#25,
                              24,
                              26,
                              32,
                              27,
                              28,
                              30,
                              5
]

line_style_palette_homogen = [1,
                              1,
                              1,
                              1,
                              1]

# line_style_palette_homogen = [1,
#                               9,
#                               7,
#                               2,
#                              3]
# line_style_palette_heterogen = [10,
#                                 5,
#                                 4,
#                                 8,
#                                 6]
line_style_palette_heterogen = [1,
                                1,
                                4,
                                8,
                                6]

fill_style_palette_left = [3305,
                           3315,
                           3325,
                           3335,
                           3345,
                           3365,
                           3375,
                           3385
]
fill_style_palette_right = [3359,
                           3351,
                           3352,
                           3353,
                           3354,
                           3356,
                           3357,
                           3358
]


class PlotableObject():
    def __init__(self, plot_object = None, is_ref = True, ref_id = -1, label = "", cuts = None, process=None, draw_option = "Marker", marker_color = 1, marker_size = 1, marker_style = 1, line_color = 1, line_width = 1, line_style = 1, fill_color = 0, fill_style = 0):
        self.plot_object = plot_object
        self.is_ref = is_ref
        self.ref_id = ref_id
        self.label = label
        self.cuts = cuts
        self.process = process
        self.draw_option = draw_option
        self.draw = draw_option
        self.marker_color = marker_color
        self.marker_size = marker_size
        self.marker_style = marker_style
        self.line_color = line_color
        self.line_width = line_width
        self.line_style = line_style
        self.fill_color = fill_color
        self.fill_style = fill_style
        
