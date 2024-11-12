from matplotlib.colors import LinearSegmentedColormap

colormap_wind = { 'rbg1':(144,238,144,1),
                  'rbg2':(255,250,205,1),
                  'rbg3':(250,128,114,1),
                  } 

colormap_wave = { 'rbg1':(230,255,255,1),
                  'rbg2':(113,180,255,1),
                  'rbg3':(7,49,255,1),
                  }

colormap_tp = { 'rbg1':(255,255,255,1),
                'rbg2':(255,227,227,1),
                'rbg3':(255,85,85,1),
                }
c1=[n / 255.0 for n in colormap_wind['rbg1'][:3]]
c2=[n / 255.0 for n in colormap_wind['rbg2'][:3]]
c3=[n / 255.0 for n in colormap_wind['rbg3'][:3]]

colorsmap = (c1,c2,c3)
cmapwind = LinearSegmentedColormap.from_list("mycmap", colorsmap)
cmapwind.set_bad('white')
vmin=-10
vmax=35

c1=tuple(list(colormap_wind['rbg1'][:-1]) + [colormap_wind['rbg1'][-1] / 2.0])
c2=tuple(list(colormap_wind['rbg2'][:-1]) + [colormap_wind['rbg2'][-1] / 2.0])
c3=tuple(list(colormap_wind['rbg3'][:-1]) + [colormap_wind['rbg3'][-1] / 2.0])

colorsmap = (c1,c2,c3)
cmapwind_p = LinearSegmentedColormap.from_list("mycmap", colorsmap)
cmapwind_p.set_bad('white')

c1=[n / 255.0 for n in colormap_wave['rbg1'][:3]]
c2=[n / 255.0 for n in colormap_wave['rbg2'][:3]]
c3=[n / 255.0 for n in colormap_wave['rbg3'][:3]]
colorsmap = (c1,c2,c3)

cmapwave=LinearSegmentedColormap.from_list("mycmap", colorsmap)
cmapwave.set_bad('white')

c1=[n / 255.0 for n in colormap_tp['rbg1'][:3]]
c2=[n / 255.0 for n in colormap_tp['rbg2'][:3]]
c3=[n / 255.0 for n in colormap_tp['rbg3'][:3]]
colorsmap = (c1,c2,c3)

cmaptp=LinearSegmentedColormap.from_list("mycmap", colorsmap)
cmaptp.set_bad('white')


