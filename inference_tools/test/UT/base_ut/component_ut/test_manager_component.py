import unittest
from unittest.mock import MagicMock

from msit.base.component.manager import BaseComponent, Component, ConsumerComp, ProducerComp, Scheduler
from msit.utils.constants import MsgConst
from msit.utils.exceptions import MsitException


class TestBaseComponent:
    def test_initialization(self):
        component = BaseComponent(priority=200)
        assert component.priority == 200
        assert component.is_activated is False

    def test_do_activate(self):
        component = BaseComponent()
        component.activate = MagicMock()
        assert component.is_activated is False

        component.do_activate()
        assert component.is_activated is True
        component.activate.assert_called_once()

        component.do_activate()
        component.activate.assert_called_once()

    def test_do_deactivate(self):
        component = BaseComponent()
        component.deactivate = MagicMock()
        component.activated = True
        assert component.is_activated is True

        component.do_deactivate()
        assert component.is_activated is False
        component.deactivate.assert_called_once()

        component.do_deactivate()
        component.deactivate.assert_called_once()

    def test_activate_does_not_change_state_directly(self):
        component = BaseComponent()
        component.activate()
        assert component.is_activated is False

    def test_deactivate_does_not_change_state_directly(self):
        component = BaseComponent()
        component.activated = True
        component.deactivate()
        assert component.is_activated is True


class ConcreteProducerComp(ProducerComp):
    def __init__(self, priority):
        super(ConcreteProducerComp, self).__init__(priority)
        self._data_generated = False

    def load_data(self):
        if not self._data_generated:
            self._data_generated = True
            return "generated_data"
        return None


class ConcreteConsumerComp(ConsumerComp):
    def __init__(self, priority):
        super(ConcreteConsumerComp, self).__init__(priority)

    def consume(self, packages):
        print("Consuming data:", packages)


class HybridComp(ProducerComp, ConsumerComp):
    def __init__(self, priority):
        super(HybridComp, self).__init__(priority)
        self._data_generated = False

    def load_data(self):
        if not self._data_generated:
            self._data_generated = True
            return "generated_data"
        return None

    def consume(self, packages):
        print("Consuming:", packages)


class TestProducerComp(unittest.TestCase):
    def setUp(self):
        self.producer = ConcreteProducerComp(priority=100)
        self.scheduler_mock = MagicMock()
        self.producer.scheduler = self.scheduler_mock

        self.producer.activate = MagicMock()
        self.producer.deactivate = MagicMock()
        self.producer.publish = MagicMock()

    def test_do_activate(self):
        self.assertFalse(self.producer.is_activated)
        self.producer.do_activate()
        self.assertTrue(self.producer.is_activated)
        self.producer.activate.assert_called_once()

    def test_do_deactivate(self):
        self.producer.activated = True
        self.producer.do_deactivate()
        self.assertFalse(self.producer.is_activated)
        self.producer.deactivate.assert_called_once()

    def test_retrieve(self):
        self.producer.publish("some_data", msg_id=1)
        self.assertIsNone(self.producer.output_buffer)

    def test_do_load_data_when_output_buffer_is_none(self):
        self.producer.load_data = MagicMock(return_value="generated_data")
        self.producer.do_load_data()
        self.producer.load_data.assert_called_once()
        self.producer.publish.assert_called_once_with("generated_data")

    def test_do_load_data_when_output_buffer_is_not_none(self):
        self.producer.output_buffer = ["some_data"]
        self.producer.do_load_data()
        self.producer.publish.assert_not_called()


class TestConsumerComp(unittest.TestCase):
    def setUp(self):
        self.producer = ConcreteProducerComp(priority=1)
        self.consumer = ConcreteConsumerComp(priority=2)
        self.comp_a = HybridComp(priority=100)
        self.comp_b = HybridComp(priority=200)
        self.comp_c = HybridComp(priority=300)
        self.consumer.consume = MagicMock()

    def test_do_consume_with_empty_dependencies(self):
        self.consumer.dependencies = {MagicMock(): None}
        self.consumer.do_consume()
        self.consumer.consume.assert_not_called()

    def test_do_consume_with_filled_dependencies(self):
        mock_producer = MagicMock()
        package_data = [mock_producer, "mock_data", 1]

        self.consumer.dependencies = {mock_producer: package_data}
        self.consumer.do_consume()

        self.consumer.consume.assert_called_once_with([package_data])
        self.assertEqual(self.consumer.dependencies[mock_producer], None)

    def test_do_consume_partial_dependencies(self):
        mock_producer1 = MagicMock()
        mock_producer2 = MagicMock()
        package_data = [mock_producer1, "mock_data", 1]

        self.consumer.dependencies = {mock_producer1: package_data, mock_producer2: None}
        self.consumer.do_consume()
        self.consumer.consume.assert_not_called()

    def test_subscribe_valid(self):
        self.consumer.subscribe(self.producer)
        self.assertIn(self.consumer, self.producer.get_subscribers())

    def test_subscribe_invalid_type(self):
        with self.assertRaises(MsitException):
            self.consumer.subscribe(self.consumer)

    def test_no_cycle(self):
        self.comp_a.subscribe(self.comp_b)
        self.comp_b.subscribe(self.comp_c)
        try:
            self.comp_c.subscribe(self.comp_a)
            self.assertTrue(True)
        except MsitException as e:
            self.fail(f"Unexpected cycle detection exception: {e}")

    def test_already_subscribed(self):
        self.comp_a.subscribe(self.comp_b)
        self.comp_b.subscribe(self.comp_c)
        self.comp_c.subscribe(self.comp_a)
        self.assertEqual(len(self.comp_c.dependencies), 1)

    def test_multiple_cycles(self):
        self.comp_a.subscribe(self.comp_b)
        self.comp_b.subscribe(self.comp_c)
        self.comp_c.subscribe(self.comp_a)
        with self.assertRaises(MsitException) as context:
            self.comp_a.subscribe(self.comp_c)
        self.assertIn(MsgConst.RISK_ALERT, str(context.exception))

    def test_on_receive(self):
        package = [self.producer, "test_data", 0]
        self.consumer.on_receive(package)
        self.assertEqual(self.consumer.dependencies[self.producer], package)

    def test_get_empty_dependencies(self):
        self.consumer.subscribe(self.producer)
        self.assertIn(self.producer, self.consumer.get_empty_dependencies())

    def test_do_consume(self):
        self.consumer.subscribe(self.producer)
        package = [self.producer, "test_data", 0]
        self.consumer.on_receive(package)


class TestRegisterDecorator(unittest.TestCase):
    def setUp(self):
        Component._component_type_map = {}

    def test_register_decorator(self):
        @Component.register("ComponentB")
        class ComponentB:
            pass

        self.assertIn("ComponentB", Component._component_type_map)
        self.assertEqual(Component._component_type_map["ComponentB"], ComponentB)

    def test_get_registered_component(self):
        @Component.register("ComponentC")
        class ComponentC:
            pass

        component = Component.get("ComponentC")
        self.assertEqual(component, ComponentC)


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.producer = MagicMock(ProducerComp)
        self.consumer = MagicMock(ConsumerComp)
        self.producer.is_ready = True
        self.consumer.is_activated = False
        self.consumer.get_empty_dependencies.return_value = []
        self.consumer.do_consume = MagicMock()

    def test_add_component(self):
        self.scheduler.add([self.producer])
        self.assertIn(self.producer, self.scheduler.comp_ref)
        self.assertEqual(self.scheduler.comp_ref[self.producer], 1)

    def test_remove_component(self):
        self.scheduler.add([self.producer])
        self.scheduler.remove([self.producer])
        self.assertNotIn(self.producer, self.scheduler.comp_ref)

    def test_schedule_consumer_when_no_dependencies(self):
        self.scheduler._schedule_consumer(self.consumer)
        self.consumer.do_consume.assert_called_once()
        self.assertIn(self.consumer, self.scheduler.comps_to_schedule)

    def test_schedule_consumer_with_unready_dependencies(self):
        dependency_mock = MagicMock()
        dependency_mock.is_ready = False
        dependency_mock.do_load_data = MagicMock()

        self.consumer.get_empty_dependencies.return_value = [dependency_mock]
        self.scheduler._schedule_consumer(self.consumer)
        self.consumer.do_consume.assert_not_called()
        dependency_mock.do_load_data.assert_called_once()
        self.assertNotIn(dependency_mock, self.scheduler.comps_to_schedule)

    def test_schedule_consumer_with_ready_dependencies(self):
        dependency_mock = MagicMock()
        dependency_mock.is_ready = True
        dependency_mock.do_load_data = MagicMock()

        self.consumer.get_empty_dependencies.return_value = [dependency_mock]
        self.scheduler._schedule_consumer(self.consumer)
        dependency_mock.do_load_data.assert_called_once()
        self.assertIn(dependency_mock, self.scheduler.comps_to_schedule)
