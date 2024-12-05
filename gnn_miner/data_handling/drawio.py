import xml.etree.ElementTree as ET
from colour import Color

class DrawIO:
  def __init__(self, filename):
    self.open(filename)

  def open(self, filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    self.models = []
    for diagram in root:
      if diagram.tag == 'diagram':
        for model in diagram:
          if model.tag == 'mxGraphModel':
            for root in model:
              if root.tag == 'root':
                self.models.append({'places': [], 'transitions': [], 'arcs': []})
                for element in root:
                  if element.tag == 'mxCell':
                    if (parsed := self.parse_element(element)) is not None:
                      self.models[-1][f'{parsed[0]["type"]}s'].extend(parsed)

  def parse_element(self, element):
    if element.attrib.get('edge', '0') == '1':
      style = self.__parse_style(element.attrib.get('style', ''))
      arrows = [{'type': 'arc', 'from': element.attrib.get('source', None), 'to': element.attrib.get('target', None)}]
      if style.get('startArrow', '') == 'classic':
        arrows.append({'type': 'arc', 'from': element.attrib.get('target', None), 'to': element.attrib.get('source', None)})
      return arrows
    elif element.attrib.get('vertex', '0') == '1':
      style = self.__parse_style(element.attrib.get('style', ''))
      if style.get('rounded', '1') == '0' or 'ellipse' not in style:
        # TODO parse/remove html style stuff in name
        return [{'type': 'transition',
                'id': element.attrib['id'],
                'name': None if element.attrib['value'] == '' else element.attrib['value']}]
      elif style.get('ellipse', False):
        if element.attrib.get('value', '') != '':
          print(element.attrib['id'], element.attrib['value'])
        return [{'type': 'place',
                'id': element.attrib['id'],
                'initial_marking': self.__classify_color(style.get('fillColor', '#ffffff')) == 'green',
                'final_marking': int(style.get('strokeWidth', 1)) > 1 or self.__classify_color(style.get('fillColor', '#ffffff')) == 'red'}]
    return None

  def __classify_color(self, hex_color):
    hsl_color = Color(hex_color).hsl
    if hsl_color[2] > 0.9:
      return 'white'
    if hsl_color[0] < 0.125 or hsl_color[0] > 0.875:
      return 'red'
    if (1/6) < hsl_color[0] < 0.5:
      return 'green'
    return 'white'

  def __parse_style(self, style_string):
    return {**{element.split('=')[0]: element.split('=')[1] for element in style_string.split(';') if '=' in element},
            **{element: True for element in style_string.split(';') if '=' not in element and element != ''}}
