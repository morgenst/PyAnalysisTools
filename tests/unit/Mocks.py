from mock import MagicMock, Mock

hist = Mock()
hist.GetName = MagicMock(return_value='foo')
