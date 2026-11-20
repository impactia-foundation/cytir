"""Task 2080: the news modules mint url-addressed ids through `cytir.core.common.mint_card_point_id`.

Proof that the swap moved no id: the helper equals the expression the modules used
(`str(uuid.uuid5(uuid.NAMESPACE_URL, url))`) on 24 URLs; EXPECTED holds the ids that expression gave
(computed from the stdlib before the swap)."""
from cytir.core.common import mint_card_point_id

EXPECTED = {
    'https://www.srf.ch/news/schweiz/a-1': '3cbc5a9a-91d8-5b43-88a5-1d93c71c44c8',
    'https://www.nzz.ch/zuerich/x-ld.1': '399af154-30d4-5e83-9582-0b8a722c4f2f',
    'https://www.rts.ch/info/suisse/1.html': '70d61b39-7769-56dd-ad7d-a365d9f44aab',
    'https://www.rsi.ch/info/svizzera/2.html': 'f7af4a85-7751-5e06-baeb-6616787b4f7e',
    'https://www.swissinfo.ch/eng/politics/3': 'c6f39f4f-89bc-5ecb-8f12-49729e54719a',
    'https://www.admin.ch/gov/de/start/dokumentation/medienmitteilungen.msg-id-1.html': '9299d8bf-a98b-52e0-b3ea-5cf8ed45c344',
    'https://example.com': '4fd35a71-71ef-5a55-a9d9-aa75c889a6d0',
    'https://example.com/': 'dd2c1780-811a-5296-81c5-178a0ef488bc',
    'http://Example.com/A?b=1&a=2#frag': 'a964002a-f69c-5622-826c-6c785077f1f3',
    'https://münchen.de/Pfad': '7e5e9050-e29e-5e03-807c-e3610b09cc99',
    'https://example.com/a%20b': '12b71b9a-6792-54b4-92eb-481305faeeec',
    'https://example.com/ä': 'afdef7ba-7312-5e42-ac2d-ed60cef58139',
    'https://example.com:8443/x': 'bb98167f-0aec-5eb5-aa30-0e814130055e',
    'https://example.com/x?utm_source=a': 'f3dac384-3ecd-5e0b-a893-77f4366ed1de',
    'https://a.b/c/d/e/f/g/h': '59efa38a-dcaf-51a3-b478-198baa007e00',
    'https://news.example.org/2026/10/31/story': 'd6294b37-2b60-5a55-976a-afddfaa60c6f',
    'https://www.watson.ch/!1': '87ee7c07-cb00-5bda-a673-25c3eb66842c',
    'https://www.blick.ch/politik/1': '58893625-2eab-5e3a-9c91-79de47e08462',
    'https://www.tagesanzeiger.ch/x-1': '24febce3-506e-5901-a4e6-32232941c406',
    'https://www.letemps.ch/suisse/1': 'bfb151c4-0624-5f74-a171-1445cd3b9841',
    'https://www.24heures.ch/1': 'b504ed28-fa7d-572f-9c60-3f457eb2c9fb',
    'https://www.20min.ch/story/1': '33fedced-e4f1-56d4-b7b1-730d72cbde2e',
    'https://www.swissinfo.ch/fre/1': 'a575b91e-e1c8-5fa2-93a8-6279c31e751c',
    '': '1b4db7eb-4057-5ddf-91e0-36dec72071f5',
}


def test_equals_the_old_inline_expression():
    assert len(EXPECTED) >= 20
    for url, old_id in EXPECTED.items():
        assert mint_card_point_id(url) == old_id
