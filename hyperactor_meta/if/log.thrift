include "thrift/annotation/thrift.thrift"

@thrift.Uri{value = "facebook.com/monarch/hyperactor_meta/thrift/MessageData"}
struct MessageData {
  // the serialized message data
  1: binary data;
}
