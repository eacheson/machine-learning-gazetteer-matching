'''
@author: eacheson
'''

from arcgis.gis import GIS
import arcgis.geometry as arcgeo

# create an arcgis 'gis' now so we can use the geometry methods
gis = GIS()

# projection codes: LV03 is wkid 21781, LV95 is 2056, wgs84 is 4326

def swiss_to_latlon(swissPoint):
    """
    Takes an [E, N] (x, y) Lv03 point and returns a [lat, lon] (y, x) point.
    """
    result_list = arcgeo.project([arcgeo.Point({"x" : swissPoint[0], "y" : swissPoint[1]})], '21781', '4326')
    result = result_list[0]
    return [result['y'], result['x']]

# this runs ~40x faster than calling the function repeatedly on each point
def swiss_to_latlon_batch(swissPoints):
    """
    Takes a list of [E, N] (x, y) Lv03 points and returns a list of [lat, lon] (y, x) points.
    """
    chpts = [arcgeo.Point({"x" : swissPoint[0], "y" : swissPoint[1]}) for swissPoint in swissPoints]
    result_list = arcgeo.project(chpts, '21781', '4326')
    return [[result['y'], result['x']] for result in result_list]

def latlon_to_swiss(latlonPoint):
    """
    Takes a [lat, lon] (y, x) point and returns an [E, N] (x, y) Lv03 point.
    """
    result_list = arcgeo.project([arcgeo.Point({"x" : latlonPoint[1], "y" : latlonPoint[0]})], '4326', '21781')
    result = result_list[0]
    return [result['x'], result['y']]

# this runs ~40x faster than calling the function repeatedly on each point
def latlon_to_swiss_batch(latlonPoints):
    """
    Takes a list of [lat, lon] (y, x) point sand returns aa list of  [E, N] (x, y) Lv03 points.
    """
    latlonpts = [arcgeo.Point({"x" : latlonPoint[1], "y" : latlonPoint[0]}) for latlonPoint in latlonPoints]
    result_list = arcgeo.project(latlonpts, '4326', '21781')
    return [[result['x'], result['y']] for result in result_list]