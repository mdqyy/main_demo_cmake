# Generated by the protocol buffer compiler.  DO NOT EDIT!

from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)



DESCRIPTOR = descriptor.FileDescriptor(
  name='detections.proto',
  package='doppia_protobuf',
  serialized_pb='\n\x10\x64\x65tections.proto\x12\x0f\x64oppia_protobuf\"\x1f\n\x07Point2d\x12\t\n\x01x\x18\x01 \x02(\x05\x12\t\n\x01y\x18\x02 \x02(\x05\"a\n\x03\x42ox\x12,\n\nmin_corner\x18\x01 \x02(\x0b\x32\x18.doppia_protobuf.Point2d\x12,\n\nmax_corner\x18\x02 \x02(\x0b\x32\x18.doppia_protobuf.Point2d\"\xfb\x01\n\tDetection\x12*\n\x0c\x62ounding_box\x18\x01 \x02(\x0b\x32\x14.doppia_protobuf.Box\x12>\n\x0cobject_class\x18\x02 \x02(\x0e\x32(.doppia_protobuf.Detection.ObjectClasses\x12\r\n\x05score\x18\x03 \x01(\x02\"s\n\rObjectClasses\x12\x07\n\x03\x43\x61r\x10\x02\x12\x0e\n\nPedestrian\x10\x03\x12\x08\n\x04\x42ike\x10\x05\x12\r\n\tMotorbike\x10\x06\x12\x07\n\x03\x42us\x10\x07\x12\x08\n\x04Tram\x10\x08\x12\x10\n\x0cStaticObject\x10\x04\x12\x0b\n\x07Unknown\x10\x00\"P\n\nDetections\x12\x12\n\nimage_name\x18\x01 \x01(\t\x12.\n\ndetections\x18\x02 \x03(\x0b\x32\x1a.doppia_protobuf.Detection')



_DETECTION_OBJECTCLASSES = descriptor.EnumDescriptor(
  name='ObjectClasses',
  full_name='doppia_protobuf.Detection.ObjectClasses',
  filename=None,
  file=DESCRIPTOR,
  values=[
    descriptor.EnumValueDescriptor(
      name='Car', index=0, number=2,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='Pedestrian', index=1, number=3,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='Bike', index=2, number=5,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='Motorbike', index=3, number=6,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='Bus', index=4, number=7,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='Tram', index=5, number=8,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='StaticObject', index=6, number=4,
      options=None,
      type=None),
    descriptor.EnumValueDescriptor(
      name='Unknown', index=7, number=0,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=306,
  serialized_end=421,
)


_POINT2D = descriptor.Descriptor(
  name='Point2d',
  full_name='doppia_protobuf.Point2d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='x', full_name='doppia_protobuf.Point2d.x', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='y', full_name='doppia_protobuf.Point2d.y', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=37,
  serialized_end=68,
)


_BOX = descriptor.Descriptor(
  name='Box',
  full_name='doppia_protobuf.Box',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='min_corner', full_name='doppia_protobuf.Box.min_corner', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='max_corner', full_name='doppia_protobuf.Box.max_corner', index=1,
      number=2, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=70,
  serialized_end=167,
)


_DETECTION = descriptor.Descriptor(
  name='Detection',
  full_name='doppia_protobuf.Detection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='bounding_box', full_name='doppia_protobuf.Detection.bounding_box', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='object_class', full_name='doppia_protobuf.Detection.object_class', index=1,
      number=2, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='score', full_name='doppia_protobuf.Detection.score', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DETECTION_OBJECTCLASSES,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=170,
  serialized_end=421,
)


_DETECTIONS = descriptor.Descriptor(
  name='Detections',
  full_name='doppia_protobuf.Detections',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    descriptor.FieldDescriptor(
      name='image_name', full_name='doppia_protobuf.Detections.image_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    descriptor.FieldDescriptor(
      name='detections', full_name='doppia_protobuf.Detections.detections', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=423,
  serialized_end=503,
)

_BOX.fields_by_name['min_corner'].message_type = _POINT2D
_BOX.fields_by_name['max_corner'].message_type = _POINT2D
_DETECTION.fields_by_name['bounding_box'].message_type = _BOX
_DETECTION.fields_by_name['object_class'].enum_type = _DETECTION_OBJECTCLASSES
_DETECTION_OBJECTCLASSES.containing_type = _DETECTION;
_DETECTIONS.fields_by_name['detections'].message_type = _DETECTION
DESCRIPTOR.message_types_by_name['Point2d'] = _POINT2D
DESCRIPTOR.message_types_by_name['Box'] = _BOX
DESCRIPTOR.message_types_by_name['Detection'] = _DETECTION
DESCRIPTOR.message_types_by_name['Detections'] = _DETECTIONS

class Point2d(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _POINT2D
  
  # @@protoc_insertion_point(class_scope:doppia_protobuf.Point2d)

class Box(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _BOX
  
  # @@protoc_insertion_point(class_scope:doppia_protobuf.Box)

class Detection(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DETECTION
  
  # @@protoc_insertion_point(class_scope:doppia_protobuf.Detection)

class Detections(message.Message):
  __metaclass__ = reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _DETECTIONS
  
  # @@protoc_insertion_point(class_scope:doppia_protobuf.Detections)

# @@protoc_insertion_point(module_scope)
